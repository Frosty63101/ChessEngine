"""
Alpha-beta chess engine with:
- TT (transposition table)
- Quiescence
- History heuristic
- Killer move heuristic
- Aspiration windows
- Late Move Reductions (LMR)
- Null-move pruning
- SEE-based capture ordering (NEW)
- PV tracking (NEW)
- Optional opening book bias from SQLite DB built by your script (NEW)
"""

import math
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum

import chess
import chess.polyglot

from model import EvalModel


class TTFlag(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: TTFlag
    bestMove: chess.Move | None


@dataclass
class SearchResult:
    bestMove: chess.Move | None
    bestScore: float
    nodes: int
    depth: int
    pvLine: list[chess.Move]


class OpeningBookDb:
    """
    Thread-safe-ish SQLite access by giving each thread its own connection.

    Why:
      - sqlite3 connections are not allowed across threads by default.
      - Your GUI calls search in a worker thread, so book lookups happen there.
    """

    def __init__(self, dbPath: str):
        self.dbPath = dbPath
        self.localState = threading.local()

    def _getConn(self) -> sqlite3.Connection:
        """
        Returns a connection specific to the current thread.
        Creates it on first use in that thread.
        """
        conn = getattr(self.localState, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.dbPath)
            conn.row_factory = sqlite3.Row

            # Optional speed pragmas for read-heavy usage
            # WAL is fine for read; if DB is already WAL from build, this is harmless.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")

            self.localState.conn = conn
        return conn

    def close(self) -> None:
        """
        Closes the current thread's connection (if any).
        Note: You *can't* easily close every thread's conn from here without bookkeeping.
        That's okay for a GUI app; connections die when threads exit / process exits.
        """
        conn = getattr(self.localState, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self.localState.conn = None

    def getMoveStats(self, positionKey: str, limit: int = 12) -> list[dict]:
        conn = self._getConn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              moveUci, san, count, whiteWins, draws, blackWins,
              sumWhiteElo, sumBlackElo
            FROM moves
            WHERE positionKey = ?
            ORDER BY count DESC
            LIMIT ?;
            """,
            (positionKey, limit),
        )
        rows = cur.fetchall()

        results: list[dict] = []
        for r in rows:
            count = int(r["count"])
            sumWhiteElo = int(r["sumWhiteElo"])
            sumBlackElo = int(r["sumBlackElo"])

            avgElo = 0
            if count > 0:
                avgElo = int((sumWhiteElo + sumBlackElo) / (2 * count))

            results.append(
                {
                    "moveUci": r["moveUci"],
                    "san": r["san"],
                    "count": count,
                    "whiteWins": int(r["whiteWins"]),
                    "draws": int(r["draws"]),
                    "blackWins": int(r["blackWins"]),
                    "avgElo": avgElo,
                }
            )
        return results


class AlphaBetaEngine:
    def __init__(self, model: EvalModel, qMaxDepth: int = 6, openingDbPath: str | None = None):
        self.model = model
        self.qMaxDepth = qMaxDepth

        self.nodeCount = 0

        self.killerMoves: list[list[chess.Move | None]] = [
            [None, None] for _ in range(256)]

        self.historyTable: dict[tuple[bool, str], int] = {}

        self.aspirationStartWindow = 0.50
        self.aspirationMaxWindow = 20.0

        self.openingDb: OpeningBookDb | None = None
        if openingDbPath is not None:
            self.openingDb = OpeningBookDb(openingDbPath)

        self.bookBiasCache: OrderedDict[str, dict[str, int]] = OrderedDict()
        self.bookBiasCacheMax = 50_000  # tweak: 10k–200k depending on RAM

        self.tt: OrderedDict[int, TTEntry] = OrderedDict()
        self.ttMaxEntries = 2_000_000  # tune: 200k–3M depending on RAM

        self.pieceValues = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }

    def close(self) -> None:
        if self.openingDb is not None:
            self.openingDb.close()
            self.openingDb = None

    def resetCaches(self) -> None:
        self.tt.clear()
        self.bookBiasCache.clear()
        self.historyTable.clear()
        for i in range(len(self.killerMoves)):
            self.killerMoves[i][0] = None
            self.killerMoves[i][1] = None

    def resetSearchHeuristics(self) -> None:
        self.historyTable.clear()
        for i in range(len(self.killerMoves)):
            self.killerMoves[i][0] = None
            self.killerMoves[i][1] = None

    def getKey(self, board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board)

    def evaluate(self, board: chess.Board) -> float:
        """
        Return evaluation from WHITE POV (positive = good for White).

        This assumes model.predict(board) is already White POV.
        """
        return float(self.model.predict(board))

    def terminalScore(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1000.0 if board.turn == chess.WHITE else 1000.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        return self.evaluate(board)

    def _ttGet(self, key: int) -> TTEntry | None:
        entry = self.tt.get(key)
        if entry is None:
            return None
        self.tt.move_to_end(key)
        return entry

    def _ttPut(self, key: int, entry: TTEntry) -> None:
        self.tt[key] = entry
        self.tt.move_to_end(key)
        while len(self.tt) > self.ttMaxEntries:
            self.tt.popitem(last=False)

    def normalizeFen4(self, board: chess.Board) -> str:
        fenParts = board.fen().split(" ")
        return " ".join(fenParts[:4])

    def moveGivesCheckSafe(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Safer than board.gives_check(move) because we control push/pop.
        Also prevents hard crashes if something slips in that isn't legal.
        """
        try:
            board.push(move)
        except AssertionError:
            return False
        isCheck = board.is_check()
        board.pop()
        return isCheck

    def isQuietMove(self, board: chess.Board, move: chess.Move) -> bool:
        if move.promotion is not None:
            return False
        if board.is_capture(move):
            return False
        if self.moveGivesCheckSafe(board, move):
            return False
        return True

    def _bookBiasCacheGet(self, positionKey: str) -> dict[str, int] | None:
        cached = self.bookBiasCache.get(positionKey)
        if cached is None:
            return None
        # mark as recently used
        self.bookBiasCache.move_to_end(positionKey)
        return cached

    def _bookBiasCachePut(self, positionKey: str, biasMap: dict[str, int]) -> None:
        self.bookBiasCache[positionKey] = biasMap
        self.bookBiasCache.move_to_end(positionKey)

        # evict least-recently-used
        while len(self.bookBiasCache) > self.bookBiasCacheMax:
            self.bookBiasCache.popitem(last=False)

    def addKiller(self, ply: int, move: chess.Move) -> None:
        if ply < 0 or ply >= len(self.killerMoves):
            return
        slot0, slot1 = self.killerMoves[ply]
        if slot0 == move:
            return
        self.killerMoves[ply][1] = slot0
        self.killerMoves[ply][0] = move

    def addHistory(self, board: chess.Board, move: chess.Move, depth: int) -> None:
        key = (board.turn, move.uci())
        bump = depth * depth
        self.historyTable[key] = self.historyTable.get(key, 0) + bump

    def getHistoryScore(self, board: chess.Board, move: chess.Move) -> int:
        return self.historyTable.get((board.turn, move.uci()), 0)

    def isLikelyEndgame(self, board: chess.Board) -> bool:
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + \
            len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + \
            len(board.pieces(chess.ROOK, chess.BLACK))
        minors = (
            len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) +
            len(board.pieces(chess.KNIGHT, chess.WHITE)) +
            len(board.pieces(chess.KNIGHT, chess.BLACK))
        )
        return queens == 0 and (rooks + minors) <= 4

    def getLeastValuableAttacker(self, board: chess.Board, attackers: chess.SquareSet, color: bool) -> tuple[int, int] | None:
        """
        Given a set of attacker squares, pick the square with the least valuable attacking piece.
        Returns (square, pieceType) or None.
        """
        bestSquare = None
        bestPieceType = None
        bestValue = 10**9

        for sq in attackers:
            piece = board.piece_at(sq)
            if piece is None or piece.color != color:
                continue
            pieceType = piece.piece_type
            value = self.pieceValues.get(pieceType, 10**6)
            if value < bestValue:
                bestValue = value
                bestSquare = sq
                bestPieceType = pieceType

        if bestSquare is None or bestPieceType is None:
            return None
        return (bestSquare, bestPieceType)

    def seeScore(self, board: chess.Board, move: chess.Move) -> int:
        """
        A simple SEE implementation:
        - Simulates the capture sequence on move.to_square with alternating least valuable attackers.
        - Returns net material gain in centipawns (positive = good capture for side-to-move).
        Notes:
        - This is an approximation (good enough for ordering).
        - We only call this for captures.
        """
        if not board.is_capture(move):
            return 0

        toSquare = move.to_square
        fromSquare = move.from_square
        usColor = board.turn
        themColor = not usColor

        victimValue = 0
        victimPiece = board.piece_at(toSquare)

        if victimPiece is not None:
            victimValue = self.pieceValues.get(victimPiece.piece_type, 0)
        else:

            if board.is_en_passant(move):
                epVictimSquare = toSquare + \
                    (-8 if usColor == chess.WHITE else 8)
                epVictimPiece = board.piece_at(epVictimSquare)
                if epVictimPiece is not None:
                    victimValue = self.pieceValues.get(
                        epVictimPiece.piece_type, 0)

        attackerPiece = board.piece_at(fromSquare)
        if attackerPiece is None:
            return 0

        attackerValue = self.pieceValues.get(attackerPiece.piece_type, 0)

        startPly = len(board.move_stack)

        board.push(move)

        gains: list[int] = [victimValue]
        sideToCapture = themColor

        currentCapturedValue = attackerValue

        while True:
            attackers = board.attackers(sideToCapture, toSquare)
            pick = self.getLeastValuableAttacker(
                board, attackers, sideToCapture)
            if pick is None:
                break

            attackerSq, attackerType = pick
            attackerVal = self.pieceValues.get(attackerType, 0)

            gains.append(currentCapturedValue - gains[-1])

            recaptureMove = chess.Move(attackerSq, toSquare)
            if recaptureMove not in board.legal_moves:

                break

            board.push(recaptureMove)

            currentCapturedValue = attackerVal
            sideToCapture = not sideToCapture

            if len(gains) > 32:
                break

        while len(board.move_stack) > startPly:
            board.pop()

        for i in range(len(gains) - 2, -1, -1):
            gains[i] = max(gains[i], -gains[i + 1])

        return gains[0]

    def _computeBookBiasMap(self, board: chess.Board, ply: int, limit: int = 20) -> dict[str, int]:
        """
        Returns moveUci -> bias for this position.
        Uses an engine-wide LRU cache to avoid repeated SQLite lookups.
        """
        if self.openingDb is None or ply > 6:
            return {}

        positionKey = self.normalizeFen4(board)

        cached = self._bookBiasCacheGet(positionKey)
        if cached is not None:
            return cached

        # One DB hit only on cache miss
        stats = self.openingDb.getMoveStats(positionKey, limit=limit)

        biasMap: dict[str, int] = {}
        for s in stats:
            count = s["count"]
            whiteWins = s["whiteWins"]
            blackWins = s["blackWins"]
            avgElo = s["avgElo"]

            if board.turn == chess.WHITE:
                wdlScore = (whiteWins - blackWins)
            else:
                wdlScore = (blackWins - whiteWins)

            bias = 0
            bias += min(20000, count * 50)
            bias += max(-5000, min(5000, wdlScore * 200))
            bias += min(5000, avgElo // 10)

            biasMap[s["moveUci"]] = bias

        self._bookBiasCachePut(positionKey, biasMap)
        return biasMap

    def getBookBiasFast(self, biasMap: dict[str, int], move: chess.Move) -> int:
        """
        O(1) lookup using the precomputed biasMap.
        """
        return biasMap.get(move.uci(), 0)

    def orderMoves(self, board: chess.Board, ttMove: chess.Move | None, ply: int):
        moves = list(board.legal_moves)

        bookBiasMap = self._computeBookBiasMap(board, ply, limit=20)

        killer0, killer1 = (None, None)
        if 0 <= ply < len(self.killerMoves):
            killer0, killer1 = self.killerMoves[ply]

        def scoreMove(move: chess.Move) -> int:
            score = 0

            if ttMove is not None and move == ttMove:
                score += 10_000_000

            score += self.getBookBiasFast(bookBiasMap, move)

            if move.promotion is not None:
                score += 900_000 + self.pieceValues.get(move.promotion, 0)

            if board.is_capture(move):
                see = self.seeScore(board, move)
                score += 600_000
                score += see * 20

                victimPiece = board.piece_at(move.to_square)
                attackerPiece = board.piece_at(move.from_square)
                victimValue = self.pieceValues.get(
                    victimPiece.piece_type, 0) if victimPiece else 0
                attackerValue = self.pieceValues.get(
                    attackerPiece.piece_type, 0) if attackerPiece else 0
                score += (victimValue * 10 - attackerValue)

            givesCheck = self.moveGivesCheckSafe(board, move)

            isQuiet = (
                move.promotion is None
                and (not board.is_capture(move))
                and (not givesCheck)
            )

            if givesCheck:
                score += 60_000

            if killer0 is not None and move == killer0:
                score += 40_000
            elif killer1 is not None and move == killer1:
                score += 35_000

            if isQuiet:
                score += self.getHistoryScore(board, move)

            return score

        moves.sort(key=scoreMove, reverse=True)
        return moves

    def quiescence(self, board: chess.Board, alpha: float, beta: float, qDepth: int, ply: int) -> float:
        self.nodeCount += 1
        if board.is_game_over():
            return self.terminalScore(board)

        standPat = self.evaluate(board)

        # Alpha/beta update based on side to move
        if board.turn == chess.WHITE:
            if standPat >= beta:
                return beta
            alpha = max(alpha, standPat)
        else:
            if standPat <= alpha:
                return alpha
            beta = min(beta, standPat)

        if qDepth <= 0:
            return standPat

        noisyMoves: list[chess.Move] = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion is not None or self.moveGivesCheckSafe(board, move):
                noisyMoves.append(move)

        noisyMoves.sort(
            key=lambda m: (
                1 if board.is_capture(m) else 0,
                self.seeScore(board, m) if board.is_capture(m) else 0
            ),
            reverse=True
        )

        if not noisyMoves:
            return standPat

        if board.turn == chess.WHITE:
            best = -math.inf
            for move in noisyMoves:
                board.push(move)
                best = max(best, self.quiescence(
                    board, alpha, beta, qDepth - 1, ply + 1))
                board.pop()
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        else:
            best = math.inf
            for move in noisyMoves:
                board.push(move)
                best = min(best, self.quiescence(
                    board, alpha, beta, qDepth - 1, ply + 1))
                board.pop()
                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best

    def alphaBeta(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0) -> float:
        self.nodeCount += 1

        if depth <= 0 or board.is_game_over():
            return self.quiescence(board, alpha, beta, self.qMaxDepth, ply)

        key = self.getKey(board)
        entry = self._ttGet(key)
        if entry is not None and entry.depth >= depth:
            if entry.flag == TTFlag.EXACT:
                return entry.score
            if entry.flag == TTFlag.LOWER:
                alpha = max(alpha, entry.score)
            elif entry.flag == TTFlag.UPPER:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        ttMove = entry.bestMove if entry is not None else None
        originalAlpha = alpha
        originalBeta = beta

        if depth >= 3 and (not board.is_check()) and (not self.isLikelyEndgame(board)):
            reduction = 2 + (depth // 5)
            board.push(chess.Move.null())
            nullScore = self.alphaBeta(
                board, depth - 1 - reduction, alpha, beta, ply + 1)
            board.pop()

            if board.turn == chess.WHITE:

                if nullScore >= beta:
                    return beta
            else:

                if nullScore <= alpha:
                    return alpha

        bestMove: chess.Move | None = None

        if board.turn == chess.WHITE:
            bestScore = -math.inf
            moves = self.orderMoves(board, ttMove, ply)

            for moveIndex, move in enumerate(moves):
                isQuiet = self.isQuietMove(board, move)
                isPvish = (ttMove is not None and move == ttMove)

                reduceBy = 0
                if depth >= 3 and moveIndex >= 4 and isQuiet and (not isPvish) and (not board.is_check()):
                    reduceBy = 1
                    if moveIndex >= 10 and depth >= 5:
                        reduceBy = 2

                board.push(move)

                if reduceBy > 0:
                    score = self.alphaBeta(
                        board, depth - 1 - reduceBy, alpha, beta, ply + 1)
                    if score > alpha:
                        score = self.alphaBeta(
                            board, depth - 1, alpha, beta, ply + 1)
                else:
                    score = self.alphaBeta(
                        board, depth - 1, alpha, beta, ply + 1)

                board.pop()

                if score > bestScore:
                    bestScore = score
                    bestMove = move

                if bestScore > alpha:
                    alpha = bestScore

                if alpha >= beta:
                    if isQuiet:
                        self.addKiller(ply, move)
                        self.addHistory(board, move, depth)
                    break

        else:
            bestScore = math.inf
            moves = self.orderMoves(board, ttMove, ply)

            for moveIndex, move in enumerate(moves):
                isQuiet = self.isQuietMove(board, move)
                isPvish = (ttMove is not None and move == ttMove)

                reduceBy = 0
                if depth >= 3 and moveIndex >= 4 and isQuiet and (not isPvish) and (not board.is_check()):
                    reduceBy = 1
                    if moveIndex >= 10 and depth >= 5:
                        reduceBy = 2

                board.push(move)

                if reduceBy > 0:
                    score = self.alphaBeta(
                        board, depth - 1 - reduceBy, alpha, beta, ply + 1)
                    if score < beta:
                        score = self.alphaBeta(
                            board, depth - 1, alpha, beta, ply + 1)
                else:
                    score = self.alphaBeta(
                        board, depth - 1, alpha, beta, ply + 1)

                board.pop()

                if score < bestScore:
                    bestScore = score
                    bestMove = move

                if bestScore < beta:
                    beta = bestScore

                if alpha >= beta:
                    if isQuiet:
                        self.addKiller(ply, move)
                        self.addHistory(board, move, depth)
                    break

        flag = TTFlag.EXACT
        if bestScore <= originalAlpha:
            flag = TTFlag.UPPER
        elif bestScore >= originalBeta:
            flag = TTFlag.LOWER

        self._ttPut(key, TTEntry(depth=depth, score=bestScore,
                                 flag=flag, bestMove=bestMove))
        return bestScore

    def reconstructPv(self, board: chess.Board, maxPlies: int = 32) -> list[chess.Move]:
        """
        Build a PV line by following TT bestMove from the current position.
        This is the standard easy PV approach because you already store bestMove in TT.
        """
        pvLine: list[chess.Move] = []
        seenKeys: set[int] = set()

        for _ in range(maxPlies):
            key = self.getKey(board)
            if key in seenKeys:
                break
            seenKeys.add(key)

            entry = self._ttGet(key)
            if entry is None or entry.bestMove is None:
                break

            move = entry.bestMove
            if move not in board.legal_moves:
                break

            pvLine.append(move)
            board.push(move)

        for _ in range(len(pvLine)):
            board.pop()

        return pvLine

    def searchFixedDepth(self, board: chess.Board, depth: int, alpha: float = -math.inf, beta: float = math.inf) -> SearchResult:
        self.nodeCount = 0

        key = self.getKey(board)
        entry = self._ttGet(key)
        ttMove = entry.bestMove if entry is not None else None

        bestMove = None

        if board.turn == chess.WHITE:
            bestScore = -math.inf
            for move in self.orderMoves(board, ttMove, ply=0):
                board.push(move)
                score = self.alphaBeta(board, depth - 1, alpha, beta, ply=1)
                board.pop()

                if score > bestScore:
                    bestScore = score
                    bestMove = move

                alpha = max(alpha, bestScore)
                if alpha >= beta:
                    break
        else:
            bestScore = math.inf
            for move in self.orderMoves(board, ttMove, ply=0):
                board.push(move)
                score = self.alphaBeta(board, depth - 1, alpha, beta, ply=1)
                board.pop()

                if score < bestScore:
                    bestScore = score
                    bestMove = move

                beta = min(beta, bestScore)
                if alpha >= beta:
                    break

        pvLine = self.reconstructPv(board, maxPlies=32)
        return SearchResult(bestMove=bestMove, bestScore=bestScore, nodes=self.nodeCount, depth=depth, pvLine=pvLine)

    def searchIterativeDeepening(self, board: chess.Board, maxDepth: int, timeLimitMs: int) -> SearchResult:
        startTime = time.time()
        bestSoFar = SearchResult(
            bestMove=None, bestScore=0.0, nodes=0, depth=0, pvLine=[])

        lastScore = 0.0
        window = self.aspirationStartWindow

        for depth in range(1, maxDepth + 1):
            elapsedMs = (time.time() - startTime) * 1000.0
            if elapsedMs >= timeLimitMs:
                break

            alpha = -math.inf
            beta = math.inf
            if depth >= 2:
                alpha = lastScore - window
                beta = lastScore + window

            while True:
                result = self.searchFixedDepth(
                    board, depth, alpha=alpha, beta=beta)

                if result.bestScore <= alpha:
                    window = min(self.aspirationMaxWindow, window * 2.0)
                    alpha = - \
                        math.inf if window >= self.aspirationMaxWindow else (
                            lastScore - window)
                    beta = lastScore + window
                    continue

                if result.bestScore >= beta:
                    window = min(self.aspirationMaxWindow, window * 2.0)
                    alpha = lastScore - window
                    beta = math.inf if window >= self.aspirationMaxWindow else (
                        lastScore + window)
                    continue

                bestSoFar = result
                lastScore = result.bestScore
                window = max(self.aspirationStartWindow, window * 0.75)
                break

            elapsedMs = (time.time() - startTime) * 1000.0
            if elapsedMs >= timeLimitMs:
                break

        return bestSoFar
