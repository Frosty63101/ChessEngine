"""
Feature extraction for chess position evaluation.

This version upgrades your feature set significantly:

- Adds a game-phase scalar in [0,1] (0=endgame, 1=opening-ish)
- Uses tapered (phase-blended) PSTs (opening PST + endgame PST)
- Upgrades pawn structure (islands, connected passers, pawn majorities, open/semi-open files, king shelter)
- Replaces noisy "legal move count" mobility with attack-map / per-piece-type mobility
- Adds outpost knights (supported by pawn and not chaseable by enemy pawns)
- Adds phase interactions so one linear model can behave differently by phase

IMPORTANT:
- This changes feature vector length and ordering.
- Your trainer already detects mismatch and will reset weights to zeros automatically.
"""

import chess
import numpy as np

# -----------------------------
# Piece values (used for tropism weighting only)
# -----------------------------

pieceValue = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.1,
    chess.QUEEN: 9.2,
    chess.KING: 0.0,
}

# -----------------------------
# PSTs
# We define opening-ish and endgame-ish tables and blend them by phase.
#
# NOTE: These are reasonable heuristics, not "the truth".
# The regression will learn the correct scaling weights anyway.
# -----------------------------

# Opening-ish PSTs (you already had these; keep them as opening baseline)
pawnPstOpening = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5, -10,  0,  0, -10, -5,  5,
    5, 10, 10, -20, -20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]
knightPstOpening = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,  0,  0,  0,  0, -20, -40,
    -30,  0, 10, 15, 15, 10,  0, -30,
    -30,  5, 15, 20, 20, 15,  5, -30,
    -30,  0, 15, 20, 20, 15,  0, -30,
    -30,  5, 10, 15, 15, 10,  5, -30,
    -40, -20,  0,  5,  5,  0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]
bishopPstOpening = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5, 10, 10,  5,  0, -10,
    -10,  5,  5, 10, 10,  5,  5, -10,
    -10,  0, 10, 10, 10, 10,  0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10,  5,  0,  0,  0,  0,  5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]
rookPstOpening = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]
queenPstOpening = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5,  5,  5,  5,  0, -10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0, -10,
    -10,  0,  5,  0,  0,  0,  0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]
kingPstOpening = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

# Endgame-ish PSTs (more king activity, pawns advanced, pieces centralized a bit differently)
pawnPstEndgame = [
    0,  0,  0,  0,  0,  0,  0,  0,
    10, 10, 10, 10, 10, 10, 10, 10,
    15, 15, 20, 25, 25, 20, 15, 15,
    20, 20, 25, 35, 35, 25, 20, 20,
    25, 25, 30, 40, 40, 30, 25, 25,
    30, 30, 35, 45, 45, 35, 30, 30,
    40, 40, 45, 55, 55, 45, 40, 40,
    0,  0,  0,  0,  0,  0,  0,  0
]
knightPstEndgame = [
    -40, -25, -20, -15, -15, -20, -25, -40,
    -25, -10,  0,  5,  5,  0, -10, -25,
    -20,  0, 10, 15, 15, 10,  0, -20,
    -15,  5, 15, 20, 20, 15,  5, -15,
    -15,  5, 15, 20, 20, 15,  5, -15,
    -20,  0, 10, 15, 15, 10,  0, -20,
    -25, -10,  0,  5,  5,  0, -10, -25,
    -40, -25, -20, -15, -15, -20, -25, -40
]
bishopPstEndgame = [
    -15, -10, -10, -10, -10, -10, -10, -15,
    -10,  0,  0,  5,  5,  0,  0, -10,
    -10,  0, 10, 10, 10, 10,  0, -10,
    -10,  5, 10, 15, 15, 10,  5, -10,
    -10,  5, 10, 15, 15, 10,  5, -10,
    -10,  0, 10, 10, 10, 10,  0, -10,
    -10,  0,  0,  5,  5,  0,  0, -10,
    -15, -10, -10, -10, -10, -10, -10, -15
]
rookPstEndgame = [
    0,  0,  5,  5,  5,  5,  0,  0,
    0,  5, 10, 10, 10, 10,  5,  0,
    0,  5,  5,  5,  5,  5,  5,  0,
    0,  5,  5, 10, 10,  5,  5,  0,
    0,  5,  5, 10, 10,  5,  5,  0,
    0,  5,  5,  5,  5,  5,  5,  0,
    0,  5, 10, 10, 10, 10,  5,  0,
    0,  0,  5,  5,  5,  5,  0,  0
]
queenPstEndgame = [
    -10, -5, -5, -5, -5, -5, -5, -10,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -10, -5, -5, -5, -5, -5, -5, -10
]
# Endgame king PST: encourage centralization
kingPstEndgame = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,  0,   0,   0,   0,   0,   0,  -10,
    -10,  0,  10,  10,  10,  10,  0,  -10,
    -10,  0,  10,  20,  20,  10,  0,  -10,
    -10,  0,  10,  20,  20,  10,  0,  -10,
    -10,  0,  10,  10,  10,  10,  0,  -10,
    -10,  0,   0,   0,   0,   0,  0,  -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

pstOpeningByPieceType = {
    chess.PAWN: pawnPstOpening,
    chess.KNIGHT: knightPstOpening,
    chess.BISHOP: bishopPstOpening,
    chess.ROOK: rookPstOpening,
    chess.QUEEN: queenPstOpening,
    chess.KING: kingPstOpening,
}
pstEndgameByPieceType = {
    chess.PAWN: pawnPstEndgame,
    chess.KNIGHT: knightPstEndgame,
    chess.BISHOP: bishopPstEndgame,
    chess.ROOK: rookPstEndgame,
    chess.QUEEN: queenPstEndgame,
    chess.KING: kingPstEndgame,
}


# -----------------------------
# Helpers
# -----------------------------

def mirrorSquare(square: int) -> int:
    """Mirror a square vertically (from White's perspective)."""
    fileIdx = chess.square_file(square)
    rankIdx = chess.square_rank(square)
    mirroredRankIdx = 7 - rankIdx
    return chess.square(fileIdx, mirroredRankIdx)


def countPieces(board: chess.Board, pieceType: int, color: bool) -> int:
    """Count pieces of a given type for a color."""
    return len(board.pieces(pieceType, color))


def _gamePhase01(board: chess.Board) -> float:
    """
    Return phase in [0,1]:
      - ~1.0 in opening (lots of material)
      - ~0.0 in endgame (material traded)

    This is a classic "piece phase" idea:
      phase = (sum(pieceWeightsRemaining) / sum(pieceWeightsStart))

    We ignore pawns (too many and weird) and use: N,B,R,Q.
    """
    phaseWeights = {
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
    }

    # Starting totals per side: 2N,2B,2R,1Q
    startTotalPerSide = 2 * 1 + 2 * 1 + 2 * 2 + 1 * 4  # 2+2+4+4 = 12
    startTotal = 2 * startTotalPerSide  # both sides -> 24

    remaining = 0
    for pieceType, w in phaseWeights.items():
        remaining += w * (countPieces(board, pieceType, chess.WHITE) +
                          countPieces(board, pieceType, chess.BLACK))

    if startTotal <= 0:
        return 0.5
    phase = float(remaining) / float(startTotal)
    # Clamp to [0,1]
    if phase < 0.0:
        phase = 0.0
    if phase > 1.0:
        phase = 1.0
    return phase


def _taperedPstDiff(board: chess.Board, phase01: float) -> float:
    """
    Tapered PST:
      blendedPst = phase * pstOpening + (1-phase) * pstEndgame
    and we compute white - black (black squares mirrored).
    """
    total = 0.0
    endWeight = 1.0 - phase01

    for square, piece in board.piece_map().items():
        openingPst = pstOpeningByPieceType[piece.piece_type]
        endgamePst = pstEndgameByPieceType[piece.piece_type]
        if piece.color == chess.WHITE:
            blended = phase01 * openingPst[square] + \
                endWeight * endgamePst[square]
            total += blended
        else:
            msq = mirrorSquare(square)
            blended = phase01 * openingPst[msq] + endWeight * endgamePst[msq]
            total -= blended

    # PST values are in centipawn-ish units; scale down
    return float(total) / 100.0


def _pawnFiles(board: chess.Board):
    """Return (whiteFiles[8], blackFiles[8], whitePawnSquares, blackPawnSquares)."""
    whitePawnSquares = list(board.pieces(chess.PAWN, chess.WHITE))
    blackPawnSquares = list(board.pieces(chess.PAWN, chess.BLACK))

    whiteFiles = [0] * 8
    blackFiles = [0] * 8

    for sq in whitePawnSquares:
        whiteFiles[chess.square_file(sq)] += 1
    for sq in blackPawnSquares:
        blackFiles[chess.square_file(sq)] += 1

    return whiteFiles, blackFiles, whitePawnSquares, blackPawnSquares


def _pawnIslandsCount(filesArr: list[int]) -> int:
    """Count pawn islands: contiguous file groups with >=1 pawn."""
    islands = 0
    inIsland = False
    for f in range(8):
        hasPawn = filesArr[f] > 0
        if hasPawn and not inIsland:
            islands += 1
            inIsland = True
        elif not hasPawn:
            inIsland = False
    return islands


def _isPassedPawn(board: chess.Board, pawnSq: int, pawnColor: bool, enemyPawns: list[int]) -> bool:
    """
    Passed pawn check:
      A pawn is passed if no enemy pawn exists ahead of it on same/adjacent files.
    enemyPawns: list of squares
    """
    pawnFile = chess.square_file(pawnSq)
    pawnRank = chess.square_rank(pawnSq)

    for enemySq in enemyPawns:
        ef = chess.square_file(enemySq)
        if abs(ef - pawnFile) > 1:
            continue
        er = chess.square_rank(enemySq)
        if pawnColor == chess.WHITE:
            if er > pawnRank:
                return False
        else:
            if er < pawnRank:
                return False
    return True


def _connectedPassedPawns(board: chess.Board, pawnColor: bool, pawnSquares: list[int], enemyPawnSquares: list[int]) -> int:
    """
    Count connected passed pawns (very endgame-relevant):
      - pawn is passed
      - and has an adjacent-file friendly passed pawn on same/near rank
    """
    passed = []
    for sq in pawnSquares:
        if _isPassedPawn(board, sq, pawnColor, enemyPawnSquares):
            passed.append(sq)

    passedSet = set(passed)
    connectedCount = 0
    for sq in passed:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        # look for adjacent passed pawn
        for df in (-1, 1):
            nf = f + df
            if nf < 0 or nf > 7:
                continue
            # scan nearby ranks (same or +/-1) for a passed pawn on adjacent file
            found = False
            for dr in (-1, 0, 1):
                nr = r + dr
                if nr < 0 or nr > 7:
                    continue
                nsq = chess.square(nf, nr)
                if nsq in passedSet:
                    found = True
                    break
            if found:
                connectedCount += 1
                break

    return connectedCount


def _backwardPawnsApprox(board: chess.Board, pawnColor: bool, pawnSquares: list[int], friendlyFiles: list[int]) -> int:
    """
    Approx backward pawn count (heuristic):
      A pawn is "backward-ish" if:
        - it's not passed
        - no friendly pawn on adjacent files is at same or more advanced rank (support chain)
        - and the square in front of it is controlled by enemy

    This is an approximation but correlates with eval.
    """
    enemyColor = not pawnColor
    backwardCount = 0

    for sq in pawnSquares:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        # Need adjacency support at same or advanced rank
        hasSupport = False
        for df in (-1, 1):
            nf = f + df
            if nf < 0 or nf > 7:
                continue
            if friendlyFiles[nf] <= 0:
                continue
            # Check if any friendly pawn exists on that file at >= rank (white) or <= rank (black)
            friendlyPawnSquares = board.pieces(chess.PAWN, pawnColor)
            for psq in friendlyPawnSquares:
                if chess.square_file(psq) != nf:
                    continue
                pr = chess.square_rank(psq)
                if pawnColor == chess.WHITE:
                    if pr >= r:
                        hasSupport = True
                        break
                else:
                    if pr <= r:
                        hasSupport = True
                        break
            if hasSupport:
                break

        if hasSupport:
            continue

        # Square in front
        if pawnColor == chess.WHITE:
            if r >= 7:
                continue
            frontSq = chess.square(f, r + 1)
        else:
            if r <= 0:
                continue
            frontSq = chess.square(f, r - 1)

        if board.is_attacked_by(enemyColor, frontSq):
            backwardCount += 1

    return backwardCount


def _openSemiOpenFileDiff(board: chess.Board, whiteFiles: list[int], blackFiles: list[int]) -> tuple[float, float]:
    """
    Open file: no pawns (both sides) on that file.
    Semi-open for a side: side has no pawn on file, but opponent does.

    We return (openFileDiff, semiOpenFileDiff) as floats:
      openFileDiff = (#open files beneficial to white rooks/queens) - (#open files beneficial to black)
      semiOpenFileDiff = (#semi-open for white) - (#semi-open for black)

    This is pawn-only; it’s a proxy for rook/queen activity.
    """
    openFiles = 0
    semiOpenWhite = 0
    semiOpenBlack = 0

    for f in range(8):
        w = whiteFiles[f]
        b = blackFiles[f]
        if w == 0 and b == 0:
            openFiles += 1
        elif w == 0 and b > 0:
            semiOpenWhite += 1
        elif b == 0 and w > 0:
            semiOpenBlack += 1

    # Open files "help both" so diff is 0; but we still provide openFiles as a global scalar (signed 0).
    # We return open count as symmetric "availability" in a signed way by multiplying with rook/queen material diff later
    # (but here keep it as 0-diff and provide openFilesCount separately elsewhere).
    openFileDiff = 0.0
    semiOpenDiff = float(semiOpenWhite - semiOpenBlack)
    return openFileDiff, semiOpenDiff


def _kingShelter(board: chess.Board, kingColor: bool) -> float:
    """
    Simple king shelter score:
      Count friendly pawns in the 3-file "shield" in front of king (up to 2 ranks ahead).

    White king shield squares are +1/+2 ranks; Black is -1/-2 ranks.
    """
    kingSq = board.king(kingColor)
    if kingSq is None:
        return 0.0

    kf = chess.square_file(kingSq)
    kr = chess.square_rank(kingSq)

    score = 0.0
    for df in (-1, 0, 1):
        nf = kf + df
        if nf < 0 or nf > 7:
            continue

        for step in (1, 2):
            nr = kr + step if kingColor == chess.WHITE else kr - step
            if nr < 0 or nr > 7:
                continue
            sq = chess.square(nf, nr)
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == kingColor:
                score += 1.0

    # Normalize a bit (max is 3 files * 2 ranks = 6)
    return score / 6.0


def _outpostKnights(board: chess.Board, knightColor: bool) -> int:
    """
    Outpost knight heuristic:
      A knight is an outpost if:
        - it's supported by a friendly pawn (pawn attacks its square)
        - and enemy pawns cannot attack that square (no pawn that could capture it next)

    This is a strong positional feature especially in middlegame.
    """
    enemyColor = not knightColor
    outposts = 0

    # Precompute enemy pawn squares for quick "can a pawn attack this square?"
    enemyPawnSquares = list(board.pieces(chess.PAWN, enemyColor))

    # For enemy pawn attacks:
    # - a white pawn attacks (file-1, rank+1) and (file+1, rank+1)
    # - a black pawn attacks (file-1, rank-1) and (file+1, rank-1)
    def enemyPawnAttacksSquare(targetSq: int) -> bool:
        tf = chess.square_file(targetSq)
        tr = chess.square_rank(targetSq)

        for psq in enemyPawnSquares:
            pf = chess.square_file(psq)
            pr = chess.square_rank(psq)
            if abs(pf - tf) != 1:
                continue
            if enemyColor == chess.WHITE:
                if pr + 1 == tr:
                    return True
            else:
                if pr - 1 == tr:
                    return True
        return False

    for ksq in board.pieces(chess.KNIGHT, knightColor):
        # Supported by friendly pawn?
        supported = False
        for psq in board.pieces(chess.PAWN, knightColor):
            if ksq in board.attacks(psq):
                supported = True
                break
        if not supported:
            continue

        # Not attackable by enemy pawns
        if enemyPawnAttacksSquare(ksq):
            continue

        outposts += 1

    return outposts


def _attackMapStats(board: chess.Board) -> tuple[float, float, float, float, float, float]:
    """
    Compute attack-map / mobility stats:

    Returns:
      (attackCountDiff, enemyHalfAttackDiff,
       knightMobDiff, bishopMobDiff, rookMobDiff, queenMobDiff)

    Where each mobility is sum of attacked squares from all pieces of that type for each side (white-black).
    """
    whiteAttackSquares = set()
    blackAttackSquares = set()

    # Per-type mobility totals
    whiteMob = {chess.KNIGHT: 0, chess.BISHOP: 0,
                chess.ROOK: 0, chess.QUEEN: 0}
    blackMob = {chess.KNIGHT: 0, chess.BISHOP: 0,
                chess.ROOK: 0, chess.QUEEN: 0}

    for sq, piece in board.piece_map().items():
        attacks = board.attacks(sq)
        if piece.color == chess.WHITE:
            whiteAttackSquares.update(attacks)
            if piece.piece_type in whiteMob:
                whiteMob[piece.piece_type] += len(attacks)
        else:
            blackAttackSquares.update(attacks)
            if piece.piece_type in blackMob:
                blackMob[piece.piece_type] += len(attacks)

    # Attack count diff (normalize by board size)
    attackCountDiff = float(len(whiteAttackSquares) -
                            len(blackAttackSquares)) / 64.0

    # Enemy half attacks:
    # - White "enemy half" is ranks 4..7
    # - Black "enemy half" is ranks 0..3
    def countInRanks(squaresSet: set[int], rankMin: int, rankMax: int) -> int:
        c = 0
        for s in squaresSet:
            r = chess.square_rank(s)
            if rankMin <= r <= rankMax:
                c += 1
        return c

    whiteInEnemyHalf = countInRanks(whiteAttackSquares, 4, 7)
    blackInEnemyHalf = countInRanks(blackAttackSquares, 0, 3)
    enemyHalfAttackDiff = float(whiteInEnemyHalf - blackInEnemyHalf) / 32.0

    knightMobDiff = float(
        whiteMob[chess.KNIGHT] - blackMob[chess.KNIGHT]) / 50.0
    bishopMobDiff = float(
        whiteMob[chess.BISHOP] - blackMob[chess.BISHOP]) / 70.0
    rookMobDiff = float(whiteMob[chess.ROOK] - blackMob[chess.ROOK]) / 70.0
    queenMobDiff = float(whiteMob[chess.QUEEN] - blackMob[chess.QUEEN]) / 90.0

    return attackCountDiff, enemyHalfAttackDiff, knightMobDiff, bishopMobDiff, rookMobDiff, queenMobDiff


def _spaceDiff(board: chess.Board) -> float:
    """
    Space proxy:
      Count how many squares in the "central band" (ranks 2..5) are attacked by each side.
    This tends to correlate with space/initiative without being as noisy as legal move counts.
    """
    targetSquares = []
    for rankIdx in range(2, 6):
        for fileIdx in range(8):
            targetSquares.append(chess.square(fileIdx, rankIdx))

    whiteCount = 0
    blackCount = 0
    for sq in targetSquares:
        if board.is_attacked_by(chess.WHITE, sq):
            whiteCount += 1
        if board.is_attacked_by(chess.BLACK, sq):
            blackCount += 1

    return float(whiteCount - blackCount) / float(len(targetSquares))


def _kingTropismAndAttackers(board: chess.Board) -> tuple[float, float]:
    """
    King tropism + king-zone attackers differential.

    - Tropism: weighted manhattan closeness of pieces to the enemy king
    - Attackers: how many squares in king zone are attacked
    """
    whiteKing = board.king(chess.WHITE)
    blackKing = board.king(chess.BLACK)
    if whiteKing is None or blackKing is None:
        return 0.0, 0.0

    def kingZoneSquares(kingSq: int):
        squares = [kingSq]
        kf = chess.square_file(kingSq)
        kr = chess.square_rank(kingSq)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                nf = kf + df
                nr = kr + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    squares.append(chess.square(nf, nr))
        return set(squares)

    whiteZone = kingZoneSquares(whiteKing)
    blackZone = kingZoneSquares(blackKing)

    def manhattan(a: int, b: int) -> int:
        return abs(chess.square_file(a) - chess.square_file(b)) + abs(chess.square_rank(a) - chess.square_rank(b))

    maxDist = 14
    whiteTrop = 0.0
    blackTrop = 0.0

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        w = pieceValue[piece.piece_type]
        if piece.color == chess.WHITE:
            d = manhattan(sq, blackKing)
            whiteTrop += w * (maxDist - d)
        else:
            d = manhattan(sq, whiteKing)
            blackTrop += w * (maxDist - d)

    whiteAttackers = 0
    blackAttackers = 0
    for sq in blackZone:
        if board.is_attacked_by(chess.WHITE, sq):
            whiteAttackers += 1
    for sq in whiteZone:
        if board.is_attacked_by(chess.BLACK, sq):
            blackAttackers += 1

    tropismDiff = float(whiteTrop - blackTrop) / 50.0
    attackersDiff = float(whiteAttackers - blackAttackers) / 10.0
    return tropismDiff, attackersDiff


# -----------------------------
# Material / basic diffs
# -----------------------------

def materialDiff(board: chess.Board):
    """Return piece count differentials for each non-king piece type."""
    pawnDiff = countPieces(board, chess.PAWN, chess.WHITE) - \
        countPieces(board, chess.PAWN, chess.BLACK)
    knightDiff = countPieces(board, chess.KNIGHT, chess.WHITE) - \
        countPieces(board, chess.KNIGHT, chess.BLACK)
    bishopDiff = countPieces(board, chess.BISHOP, chess.WHITE) - \
        countPieces(board, chess.BISHOP, chess.BLACK)
    rookDiff = countPieces(board, chess.ROOK, chess.WHITE) - \
        countPieces(board, chess.ROOK, chess.BLACK)
    queenDiff = countPieces(board, chess.QUEEN, chess.WHITE) - \
        countPieces(board, chess.QUEEN, chess.BLACK)
    return float(pawnDiff), float(knightDiff), float(bishopDiff), float(rookDiff), float(queenDiff)


def bishopPairDiff(board: chess.Board) -> float:
    """Return 1, 0, or -1 based on who has the bishop pair."""
    whiteBishops = countPieces(board, chess.BISHOP, chess.WHITE)
    blackBishops = countPieces(board, chess.BISHOP, chess.BLACK)
    return float(1 if whiteBishops >= 2 else 0) - float(1 if blackBishops >= 2 else 0)


# -----------------------------
# Pawn structure (upgraded)
# -----------------------------

def pawnStructureFeatures(board: chess.Board):
    """
    Compute a richer pawn structure feature bundle.

    Returns:
      passedDiff,
      isolatedDiff,
      doubledDiff,
      islandsDiff,
      connectedPassedDiff,
      backwardDiff,
      pawnMajorityDiff,
      openFilesCount,
      semiOpenFileDiff,
      kingShelterDiff
    """
    whiteFiles, blackFiles, whitePawnSquares, blackPawnSquares = _pawnFiles(
        board)

    # Doubled pawns: extra pawns beyond 1 per file
    doubledWhite = sum(max(0, c - 1) for c in whiteFiles)
    doubledBlack = sum(max(0, c - 1) for c in blackFiles)

    # Isolated pawns: pawns on file where adjacent files have zero friendly pawns
    isolatedWhite = 0
    isolatedBlack = 0

    for fileIdx in range(8):
        if whiteFiles[fileIdx] > 0:
            leftHas = whiteFiles[fileIdx -
                                 1] > 0 if fileIdx - 1 >= 0 else False
            rightHas = whiteFiles[fileIdx +
                                  1] > 0 if fileIdx + 1 < 8 else False
            if not leftHas and not rightHas:
                isolatedWhite += whiteFiles[fileIdx]

        if blackFiles[fileIdx] > 0:
            leftHas = blackFiles[fileIdx -
                                 1] > 0 if fileIdx - 1 >= 0 else False
            rightHas = blackFiles[fileIdx +
                                  1] > 0 if fileIdx + 1 < 8 else False
            if not leftHas and not rightHas:
                isolatedBlack += blackFiles[fileIdx]

    # Passed pawns
    passedWhite = 0
    passedBlack = 0
    for wp in whitePawnSquares:
        if _isPassedPawn(board, wp, chess.WHITE, blackPawnSquares):
            passedWhite += 1
    for bp in blackPawnSquares:
        if _isPassedPawn(board, bp, chess.BLACK, whitePawnSquares):
            passedBlack += 1

    # Pawn islands
    islandsWhite = _pawnIslandsCount(whiteFiles)
    islandsBlack = _pawnIslandsCount(blackFiles)

    # Connected passed pawns
    connectedPassedWhite = _connectedPassedPawns(
        board, chess.WHITE, whitePawnSquares, blackPawnSquares)
    connectedPassedBlack = _connectedPassedPawns(
        board, chess.BLACK, blackPawnSquares, whitePawnSquares)

    # Backward pawns (approx)
    backwardWhite = _backwardPawnsApprox(
        board, chess.WHITE, whitePawnSquares, whiteFiles)
    backwardBlack = _backwardPawnsApprox(
        board, chess.BLACK, blackPawnSquares, blackFiles)

    # Pawn majorities: queenside (a-c) and kingside (f-h)
    whiteQueenSide = sum(whiteFiles[0:3])
    blackQueenSide = sum(blackFiles[0:3])
    whiteKingSide = sum(whiteFiles[5:8])
    blackKingSide = sum(blackFiles[5:8])
    pawnMajorityDiff = float(
        (whiteQueenSide - blackQueenSide) - (whiteKingSide - blackKingSide)) / 8.0

    # Open/semi-open files
    openFilesCount = 0
    for f in range(8):
        if whiteFiles[f] == 0 and blackFiles[f] == 0:
            openFilesCount += 1
    _, semiOpenFileDiff = _openSemiOpenFileDiff(board, whiteFiles, blackFiles)

    # King shelter diff (white - black)
    kingShelterWhite = _kingShelter(board, chess.WHITE)
    kingShelterBlack = _kingShelter(board, chess.BLACK)
    kingShelterDiff = float(kingShelterWhite - kingShelterBlack)

    return (
        float(passedWhite - passedBlack),
        float(isolatedWhite - isolatedBlack),
        float(doubledWhite - doubledBlack),
        float(islandsWhite - islandsBlack),
        float(connectedPassedWhite - connectedPassedBlack),
        float(backwardWhite - backwardBlack),
        float(pawnMajorityDiff),
        float(openFilesCount) / 8.0,  # normalize
        float(semiOpenFileDiff) / 8.0,  # normalize
        float(kingShelterDiff),
    )


# -----------------------------
# Feature vector
# -----------------------------

def boardToFeatureVector(board: chess.Board) -> np.ndarray:
    """
    Build the full feature vector used by the evaluation model.

    Design goals:
      - Mostly stable, position-based signals (not super noisy "legal moves")
      - Phase-aware without needing multiple separate models
      - Still cheap enough to run thousands of times per second in Python
    """
    # Phase in [0,1]
    phase01 = _gamePhase01(board)
    end01 = 1.0 - phase01

    # Material diffs
    pawnDiff, knightDiff, bishopDiff, rookDiff, queenDiff = materialDiff(board)
    bishopPair = bishopPairDiff(board)

    # Pawn structure (upgraded)
    (
        passedDiff,
        isolatedDiff,
        doubledDiff,
        islandsDiff,
        connectedPassedDiff,
        backwardDiff,
        pawnMajorityDiff,
        openFilesCount,
        semiOpenFileDiff,
        kingShelterDiff,
    ) = pawnStructureFeatures(board)

    # Tapered PST diff
    pstTaperedDiff = _taperedPstDiff(board, phase01)

    # King pressure
    tropismDiff, kingAttackersDiff = _kingTropismAndAttackers(board)

    # Attack maps / per-piece mobility (more stable than legal moves)
    attackCountDiff, enemyHalfAttackDiff, knightMobDiff, bishopMobDiff, rookMobDiff, queenMobDiff = _attackMapStats(
        board)

    # Space proxy
    spaceDiff = _spaceDiff(board)

    # Outpost knights
    outpostWhite = _outpostKnights(board, chess.WHITE)
    outpostBlack = _outpostKnights(board, chess.BLACK)
    outpostDiff = float(outpostWhite - outpostBlack)

    # Tempo (side to move)
    tempo = 1.0 if board.turn == chess.WHITE else -1.0

    # Center control (keep it; cheap and useful)
    centerSquares = [chess.D4, chess.E4, chess.D5, chess.E5]
    whiteCenterAttacks = sum(board.is_attacked_by(
        chess.WHITE, sq) for sq in centerSquares)
    blackCenterAttacks = sum(board.is_attacked_by(
        chess.BLACK, sq) for sq in centerSquares)
    centerAttackDiff = float(whiteCenterAttacks - blackCenterAttacks) / 4.0

    # -----------------------------
    # Phase interaction features
    # These let a single linear model behave differently by phase.
    # -----------------------------
    passedEndgame = passedDiff * end01
    connectedPassedEndgame = connectedPassedDiff * end01
    kingAttackMidgame = kingAttackersDiff * phase01
    kingShelterMidgame = kingShelterDiff * phase01
    outpostMidgame = outpostDiff * phase01
    mobilityMidgame = (knightMobDiff + bishopMobDiff +
                       rookMobDiff + queenMobDiff) * phase01
    pawnMajorityEndgame = pawnMajorityDiff * end01
    islandsEndgame = islandsDiff * end01
    backwardMidgame = backwardDiff * phase01

    # A little “open files become more important when rooks/queens exist”
    majorPiecePresence = float(
        abs(rookDiff) + abs(queenDiff)) / 6.0  # rough scale
    openFilesWithMajors = openFilesCount * majorPiecePresence

    # -----------------------------
    # Assemble vector (bias first)
    # -----------------------------
    x = np.array([
        1.0,                    # bias (must be constant 1.0)

        # Phase scalars
        phase01,
        end01,

        # Material
        pawnDiff,
        knightDiff,
        bishopDiff,
        rookDiff,
        queenDiff,
        bishopPair,

        # Pawn structure base
        passedDiff,
        -isolatedDiff,          # non-isolated (so positive is good)
        -doubledDiff,           # non-doubled (so positive is good)
        -islandsDiff,           # fewer islands is good
        -backwardDiff,          # fewer backward is good
        connectedPassedDiff,
        pawnMajorityDiff,
        semiOpenFileDiff,
        openFilesCount,
        kingShelterDiff,

        # Positional / activity
        pstTaperedDiff,
        centerAttackDiff,
        spaceDiff,

        # King pressure
        tropismDiff,
        kingAttackersDiff,

        # Attack map / mobility
        attackCountDiff,
        enemyHalfAttackDiff,
        knightMobDiff,
        bishopMobDiff,
        rookMobDiff,
        queenMobDiff,

        # Outposts
        outpostDiff,

        # Tempo
        tempo,

        # Phase interactions (key)
        passedEndgame,
        connectedPassedEndgame,
        pawnMajorityEndgame,
        islandsEndgame,
        kingAttackMidgame,
        kingShelterMidgame,
        outpostMidgame,
        mobilityMidgame,
        backwardMidgame,
        openFilesWithMajors,
    ], dtype=np.float64)

    return x


def featureNames():
    """Return the feature names in the same order as boardToFeatureVector()."""
    return [
        "bias",

        "phase01",
        "end01",

        "pawnDiff",
        "knightDiff",
        "bishopDiff",
        "rookDiff",
        "queenDiff",
        "bishopPairDiff",

        "passedPawnDiff",
        "nonIsolatedPawnDiff",
        "nonDoubledPawnDiff",
        "lowPawnIslandsDiff",
        "lowBackwardPawnDiff",
        "connectedPassedPawnDiff",
        "pawnMajorityDiff",
        "semiOpenFileDiff",
        "openFilesCount",
        "kingShelterDiff",

        "pstTaperedDiff",
        "centerAttackDiff",
        "spaceDiff",

        "kingTropismDiff",
        "kingAttackersDiff",

        "attackCountDiff",
        "enemyHalfAttackDiff",
        "knightMobDiff",
        "bishopMobDiff",
        "rookMobDiff",
        "queenMobDiff",

        "outpostKnightDiff",

        "tempo",

        "passedEndgame",
        "connectedPassedEndgame",
        "pawnMajorityEndgame",
        "islandsEndgame",
        "kingAttackMidgame",
        "kingShelterMidgame",
        "outpostMidgame",
        "mobilityMidgame",
        "backwardMidgame",
        "openFilesWithMajors",
    ]
