"""Alpha-beta chess engine with transposition table support."""

import math
import time
from dataclasses import dataclass
from enum import IntEnum

import chess
import chess.polyglot

from model import EvalModel


class TTFlag(IntEnum):
    """Transposition-table entry type for score bounds."""

    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass
class TTEntry:
    """Cached search result for a position at a given depth."""

    depth: int
    score: float
    flag: TTFlag
    bestMove: chess.Move | None


@dataclass
class SearchResult:
    """Container for a completed search iteration."""

    bestMove: chess.Move | None
    bestScore: float
    nodes: int
    depth: int


class AlphaBetaEngine:
    """Alpha-beta engine with quiescence and transposition table."""

    def __init__(self, model: EvalModel, qMaxDepth: int = 6):
        self.model = model
        self.qMaxDepth = qMaxDepth
        self.tt: dict[int, TTEntry] = {}
        self.nodeCount = 0

    def getKey(self, board: chess.Board) -> int:
        """Return the Zobrist hash for the current board state."""
        return chess.polyglot.zobrist_hash(board)

    def evaluate(self, board: chess.Board) -> float:
        """Evaluate a position using the current model."""
        return self.model.predict(board)

    def terminalScore(self, board: chess.Board) -> float:
        """Return a terminal score or fall back to evaluation."""
        if board.is_checkmate():
            return -1000.0 if board.turn == chess.WHITE else 1000.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        return self.evaluate(board)

    def orderMoves(self, board: chess.Board, ttMove: chess.Move | None):
        """Generate legal moves ordered by tactical and TT heuristics."""
        moves = list(board.legal_moves)
        pieceValues = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
            None: 0
        }

        def scoreMove(move: chess.Move) -> int:
            """Score a move for ordering using captures, promotions, and checks."""
            score = 0
            if ttMove is not None and move == ttMove:
                score += 10_000_000
            if move.promotion is not None:
                score += 900_000 + pieceValues.get(move.promotion, 0)
            if board.is_capture(move):
                victimPiece = board.piece_at(move.to_square)
                attackerPiece = board.piece_at(move.from_square)
                victimValue = pieceValues.get(
                    victimPiece.piece_type if victimPiece else None, 0)
                attackerValue = pieceValues.get(
                    attackerPiece.piece_type if attackerPiece else None, 0)
                score += 500_000 + (victimValue * 10 - attackerValue)
            if board.gives_check(move):
                score += 50_000
            return score
        moves.sort(key=scoreMove, reverse=True)
        return moves

    def quiescence(self, board: chess.Board, alpha: float, beta: float, qDepth: int) -> float:
        """Extend search through tactical noise to stabilize evaluation."""
        self.nodeCount += 1
        if board.is_game_over():
            return self.terminalScore(board)
        standPat = self.evaluate(board)
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
        noisyMoves = []
        for move in board.legal_moves:
            if board.is_capture(move):
                noisyMoves.append(move)
            else:
                board.push(move)
                isCheck = board.is_check()
                board.pop()
                if isCheck:
                    noisyMoves.append(move)
        noisyMoves.sort(key=lambda m: 1 if board.is_capture(m)
                        else 0, reverse=True)
        if board.turn == chess.WHITE:
            best = -math.inf
            for move in noisyMoves:
                board.push(move)
                best = max(best, self.quiescence(
                    board, alpha, beta, qDepth - 1))
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
                    board, alpha, beta, qDepth - 1))
                board.pop()

                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best

    def alphaBeta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Alpha-beta search with transposition table and quiescence."""
        self.nodeCount += 1
        if depth <= 0 or board.is_game_over():
            return self.quiescence(board, alpha, beta, self.qMaxDepth)
        key = self.getKey(board)
        entry = self.tt.get(key)
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
        bestMove = None
        if board.turn == chess.WHITE:
            bestScore = -math.inf
            for move in self.orderMoves(board, ttMove):
                board.push(move)
                score = self.alphaBeta(board, depth - 1, alpha, beta)
                board.pop()
                if score > bestScore:
                    bestScore = score
                    bestMove = move
                alpha = max(alpha, bestScore)
                if alpha >= beta:
                    break
        else:
            bestScore = math.inf
            for move in self.orderMoves(board, ttMove):
                board.push(move)
                score = self.alphaBeta(board, depth - 1, alpha, beta)
                board.pop()
                if score < bestScore:
                    bestScore = score
                    bestMove = move
                beta = min(beta, bestScore)
                if alpha >= beta:
                    break
        flag = TTFlag.EXACT
        if bestScore <= originalAlpha:
            flag = TTFlag.UPPER
        elif bestScore >= originalBeta:
            flag = TTFlag.LOWER
        self.tt[key] = TTEntry(
            depth=depth,
            score=bestScore,
            flag=flag,
            bestMove=bestMove
        )
        return bestScore

    def searchFixedDepth(self, board: chess.Board, depth: int) -> SearchResult:
        """Search to a fixed depth and return the best move found."""
        self.nodeCount = 0
        key = self.getKey(board)
        entry = self.tt.get(key)
        ttMove = entry.bestMove if entry is not None else None
        bestMove = None
        alpha = -math.inf
        beta = math.inf
        if board.turn == chess.WHITE:
            bestScore = -math.inf
            for move in self.orderMoves(board, ttMove):
                board.push(move)
                score = self.alphaBeta(board, depth - 1, alpha, beta)
                board.pop()
                if score > bestScore:
                    bestScore = score
                    bestMove = move
                alpha = max(alpha, bestScore)
                if alpha >= beta:
                    break
        else:
            bestScore = math.inf
            for move in self.orderMoves(board, ttMove):
                board.push(move)
                score = self.alphaBeta(board, depth - 1, alpha, beta)
                board.pop()
                if score < bestScore:
                    bestScore = score
                    bestMove = move
                beta = min(beta, bestScore)
                if alpha >= beta:
                    break
        return SearchResult(bestMove=bestMove, bestScore=bestScore, nodes=self.nodeCount, depth=depth)

    def searchIterativeDeepening(self, board: chess.Board, maxDepth: int, timeLimitMs: int) -> SearchResult:
        """Iteratively deepen until depth or time limit is reached."""
        startTime = time.time()
        bestSoFar = SearchResult(
            bestMove=None, bestScore=0.0, nodes=0, depth=0)
        for depth in range(1, maxDepth + 1):
            elapsedMs = (time.time() - startTime) * 1000.0
            if elapsedMs >= timeLimitMs:
                break
            result = self.searchFixedDepth(board, depth)
            bestSoFar = result
            elapsedMs = (time.time() - startTime) * 1000.0
            if elapsedMs >= timeLimitMs:
                break
        return bestSoFar
