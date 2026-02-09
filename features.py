"""Feature extraction for chess position evaluation."""

import chess
import numpy as np

pawnPst = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5, -10,  0,  0, -10, -5,  5,
    5, 10, 10, -20, -20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]
knightPst = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,  0,  0,  0,  0, -20, -40,
    -30,  0, 10, 15, 15, 10,  0, -30,
    -30,  5, 15, 20, 20, 15,  5, -30,
    -30,  0, 15, 20, 20, 15,  0, -30,
    -30,  5, 10, 15, 15, 10,  5, -30,
    -40, -20,  0,  5,  5,  0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]
bishopPst = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5, 10, 10,  5,  0, -10,
    -10,  5,  5, 10, 10,  5,  5, -10,
    -10,  0, 10, 10, 10, 10,  0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10,  5,  0,  0,  0,  0,  5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]
rookPst = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]
queenPst = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5,  5,  5,  5,  0, -10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0, -10,
    -10,  0,  5,  0,  0,  0,  0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]
kingPst = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]
pstByPieceType = {
    chess.PAWN: pawnPst,
    chess.KNIGHT: knightPst,
    chess.BISHOP: bishopPst,
    chess.ROOK: rookPst,
    chess.QUEEN: queenPst,
    chess.KING: kingPst,
}
pieceValue = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.1,
    chess.QUEEN: 9.2,
    chess.KING: 0.0,
}


def mirrorSquare(square: int) -> int:
    """Mirror a square vertically (from White's perspective)."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    mirroredRank = 7 - rank
    return chess.square(file, mirroredRank)


def countPieces(board: chess.Board, pieceType: int, color: bool) -> int:
    """Count the number of pieces of a given type and color."""
    return len(board.pieces(pieceType, color))


def pawnStructureFeatures(board: chess.Board):
    """Compute passed, isolated, and doubled pawn differentials."""
    passedWhite = 0
    passedBlack = 0
    isolatedWhite = 0
    isolatedBlack = 0
    doubledWhite = 0
    doubledBlack = 0
    whitePawns = list(board.pieces(chess.PAWN, chess.WHITE))
    blackPawns = list(board.pieces(chess.PAWN, chess.BLACK))
    whiteFiles = [0]*8
    blackFiles = [0]*8
    for sq in whitePawns:
        whiteFiles[chess.square_file(sq)] += 1
    for sq in blackPawns:
        blackFiles[chess.square_file(sq)] += 1
    doubledWhite = sum(max(0, c - 1) for c in whiteFiles)
    doubledBlack = sum(max(0, c - 1) for c in blackFiles)
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
    for wp in whitePawns:
        wf = chess.square_file(wp)
        wr = chess.square_rank(wp)
        isPassed = True
        for bf in [wf - 1, wf, wf + 1]:
            if bf < 0 or bf > 7:
                continue
            for bp in blackPawns:
                if chess.square_file(bp) == bf and chess.square_rank(bp) > wr:
                    isPassed = False
                    break
            if not isPassed:
                break
        if isPassed:
            passedWhite += 1
    for bp in blackPawns:
        bf = chess.square_file(bp)
        br = chess.square_rank(bp)
        isPassed = True
        for wf in [bf - 1, bf, bf + 1]:
            if wf < 0 or wf > 7:
                continue
            for wp in whitePawns:
                if chess.square_file(wp) == wf and chess.square_rank(wp) < br:
                    isPassed = False
                    break
            if not isPassed:
                break
        if isPassed:
            passedBlack += 1
    return float(passedWhite - passedBlack), float(isolatedWhite - isolatedBlack), float(doubledWhite - doubledBlack)


def bishopPairDiff(board: chess.Board) -> float:
    """Return 1, 0, or -1 based on who has the bishop pair."""
    whiteBishops = countPieces(board, chess.BISHOP, chess.WHITE)
    blackBishops = countPieces(board, chess.BISHOP, chess.BLACK)
    return float(1 if whiteBishops >= 2 else 0) - float(1 if blackBishops >= 2 else 0)


def pstDiff(board: chess.Board) -> float:
    """Compute piece-square table differential (white minus black)."""
    total = 0
    for square, piece in board.piece_map().items():
        pst = pstByPieceType[piece.piece_type]
        if piece.color == chess.WHITE:
            total += pst[square]
        else:
            total -= pst[mirrorSquare(square)]
    return float(total) / 100.0


def kingTropismAndAttackers(board: chess.Board):
    """Compute king tropism and king-zone attacker differentials."""
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
    return float(whiteTrop - blackTrop) / 50.0, float(whiteAttackers - blackAttackers) / 10.0


def mobilitySigned(board: chess.Board) -> float:
    """Return signed mobility based on side to move."""
    m = board.legal_moves.count()
    return float(m) if board.turn == chess.WHITE else -float(m)


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


def boardToFeatureVector(board: chess.Board) -> np.ndarray:
    """Build the full feature vector used by the evaluation model."""
    pawnDiff, knightDiff, bishopDiff, rookDiff, queenDiff = materialDiff(board)
    passedDiff, isolatedDiff, doubledDiff = pawnStructureFeatures(board)
    bishopPair = bishopPairDiff(board)
    pst = pstDiff(board)
    tropismDiff, kingAttackersDiff = kingTropismAndAttackers(board)
    mob = mobilitySigned(board) / 20.0
    tempo = 1.0 if board.turn == chess.WHITE else -1.0
    centerSquares = [chess.D4, chess.E4, chess.D5, chess.E5]
    whiteCenterAttacks = sum(board.is_attacked_by(
        chess.WHITE, sq) for sq in centerSquares)
    blackCenterAttacks = sum(board.is_attacked_by(
        chess.BLACK, sq) for sq in centerSquares)
    centerAttackDiff = float(whiteCenterAttacks - blackCenterAttacks) / 4.0
    x = np.array([
        1.0,
        pawnDiff,
        knightDiff,
        bishopDiff,
        rookDiff,
        queenDiff,
        passedDiff,
        -isolatedDiff,
        -doubledDiff,
        bishopPair,
        pst,
        mob,
        centerAttackDiff,
        tropismDiff,
        kingAttackersDiff,
        tempo,
    ], dtype=np.float64)
    return x


def featureNames():
    """Return the feature names in the same order as the vector."""
    return [
        "bias",
        "pawnDiff", "knightDiff", "bishopDiff", "rookDiff", "queenDiff",
        "passedPawnDiff", "nonIsolatedPawnDiff", "nonDoubledPawnDiff",
        "bishopPairDiff", "pstDiff",
        "mobilitySigned", "centerAttackDiff",
        "kingTropismDiff", "kingAttackersDiff",
        "tempo",
    ]
