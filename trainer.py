"""
Training utilities for fitting the evaluation model to Stockfish.

Changes in this version:
1) Early-stop support: caller can provide shouldStop() callback; if it returns True,
   we stop collecting new samples and train on what we have collected so far.
2) Dupe reduction:
   - exact dedupe by positionKey() (already present, kept)
   - PGN sampling uses a shuffled pool iterator instead of random.choice() (reduces repeat attempts)
   - near-dupe reduction by feature-bucket capping (prevents dataset being flooded by similar positions)

Notes:
- Assumes features.py produces your latest feature vector and featureNames() matches it.
"""

import json
import os
import random
from typing import Callable, Optional

import chess
import chess.engine
import chess.pgn
import numpy as np
from tqdm import tqdm

from config import (baseL2Lambda, labelTimeMs, labelUseTimeMs,
                    maxPgnFilesToScan, pgnDir, pgnGamesPerFileLimit,
                    pgnPositionsPerGameLimit, randomPliesMax, selfPlayDepth,
                    selfPlayMaxPlies, selfPlayRandomness, trainMixPgn,
                    trainMixRandom, trainMixSelfPlay, weightsPath)
from engine import AlphaBetaEngine
from features import boardToFeatureVector, featureNames
from model import EvalModel, loadModel, saveModel

# -----------------------------
# Position key + dedupe helpers
# -----------------------------


def positionKey(board: chess.Board) -> str:
    """
    Stable dedupe key that ignores move counters (halfmove/fullmove).
    Uses only:
      - piece placement
      - side to move
      - castling rights
      - en passant square
    """
    placement = board.board_fen()
    turn = "w" if board.turn == chess.WHITE else "b"
    castling = board.castling_xfen()
    ep = "-" if board.ep_square is None else chess.square_name(board.ep_square)
    return f"{placement} {turn} {castling} {ep}"


# -----------------------------
# Stockfish score helpers
# -----------------------------

def stockfishScoreToPawns(score: chess.engine.PovScore) -> float:
    """
    Convert Stockfish score to pawns from White POV:
      - centipawns -> cp/100
      - mate -> +/-100 pawns
    """
    pov = score.pov(chess.WHITE)
    if pov.is_mate():
        mateIn = pov.mate()
        if mateIn is None:
            return 0.0
        return 100.0 if mateIn > 0 else -100.0

    cp = pov.score(mate_score=100000)
    if cp is None:
        return 0.0
    return float(cp) / 100.0


def debugScorePovs(infoScore: chess.engine.PovScore, board: chess.Board) -> None:
    whitePov = infoScore.pov(chess.WHITE)
    stmPov = infoScore.pov(board.turn)

    def scoreToStr(s: chess.engine.PovScore) -> str:
        if s.is_mate():
            return f"mate {s.mate()}"
        cp = s.score(mate_score=100000)
        return f"cp {cp}"

    print(
        f"[labeldbg] turn={'W' if board.turn==chess.WHITE else 'B'} "
        f"whitePOV={scoreToStr(whitePov)} stmPOV={scoreToStr(stmPov)}",
        flush=True
    )


def analyseBoardWithFish(fish: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> float:
    """
    Label a board with Stockfish and return pawn score (White POV).
    Uses time-based analysis if labelUseTimeMs is enabled.
    """
    if labelUseTimeMs and labelTimeMs > 0:
        info = fish.analyse(board, chess.engine.Limit(
            time=labelTimeMs / 1000.0))
        debugScorePovs(info["score"], board)
    else:
        info = fish.analyse(board, chess.engine.Limit(depth=depth))
    return stockfishScoreToPawns(info["score"])


# -----------------------------
# Position generation (random / self-play)
# -----------------------------

def generateRandomPosition(maxPlies: int) -> chess.Board:
    """Generate a random legal position by playing random plies from the start."""
    board = chess.Board()
    plies = random.randint(2, maxPlies)
    for _ in range(plies):
        if board.is_game_over():
            break
        legalMoves = list(board.legal_moves)
        if not legalMoves:
            break
        board.push(random.choice(legalMoves))
    return board


def generateSelfPlayPosition(engine: AlphaBetaEngine, maxPlies: int, depth: int, randomness: float) -> chess.Board:
    """
    Generate a position via self-play:
      - random move with probability `randomness`
      - otherwise use your AlphaBetaEngine at fixed depth
    """
    board = chess.Board()
    plies = random.randint(6, maxPlies)
    for _ in range(plies):
        if board.is_game_over():
            break
        legalMoves = list(board.legal_moves)
        if not legalMoves:
            break

        if random.random() < randomness:
            board.push(random.choice(legalMoves))
            continue

        result = engine.searchFixedDepth(board, depth)
        board.push(
            result.bestMove if result.bestMove is not None else random.choice(legalMoves))
    return board


# -----------------------------
# PGN sampling (opening/mid/end) + dedupe
# -----------------------------

def iterPgnFiles() -> list[str]:
    if not os.path.isdir(pgnDir):
        return []
    files = [
        os.path.join(pgnDir, f)
        for f in os.listdir(pgnDir)
        if f.lower().endswith(".pgn")
    ]
    print(f"[train] found {len(files)} PGN files in {pgnDir}", flush=True)
    return files


def materialCountExcludingKings(board: chess.Board) -> int:
    pieceMap = board.piece_map()
    total = 0
    for _, piece in pieceMap.items():
        if piece.piece_type != chess.KING:
            total += 1
    return total


def collectGamePositionsByPhase(game: chess.pgn.Game) -> tuple[list[chess.Board], list[chess.Board], list[chess.Board]]:
    openingCandidates: list[chess.Board] = []
    midCandidates: list[chess.Board] = []
    endCandidates: list[chess.Board] = []

    boardsAlongLine: list[chess.Board] = []
    board = game.board()

    ply = 0
    for move in game.mainline_moves():
        board.push(move)
        ply += 1
        if board.is_game_over():
            break
        boardsAlongLine.append(board.copy(stack=False))

    if not boardsAlongLine:
        return openingCandidates, midCandidates, endCandidates

    totalPlies = len(boardsAlongLine)

    # Opening bucket
    for idx in range(totalPlies):
        p = idx + 1
        if 8 <= p <= 22 and (p % 2 == 0):
            openingCandidates.append(boardsAlongLine[idx])

    # Middlegame bucket
    midStart = 24
    midEnd = max(midStart, totalPlies - 20)
    for idx in range(totalPlies):
        p = idx + 1
        if midStart <= p <= midEnd and (p % 4 == 0):
            midCandidates.append(boardsAlongLine[idx])

    # Endgame bucket
    endWindow = min(20, totalPlies)
    tail = boardsAlongLine[-endWindow:]
    tailSorted = sorted(tail, key=materialCountExcludingKings)
    for b in tailSorted[: min(8, len(tailSorted))]:
        endCandidates.append(b)

    return openingCandidates, midCandidates, endCandidates


def samplePgnPositions(maxPositions: int) -> list[chess.Board]:
    positions: list[chess.Board] = []
    seenKeys: set[str] = set()
    duplicatesSkipped = 0

    pgnFiles = iterPgnFiles()
    if not pgnFiles:
        print("[pgn] no PGN files found", flush=True)
        return positions

    random.shuffle(pgnFiles)
    pgnFiles = pgnFiles[:maxPgnFilesToScan]

    print(
        f"[pgn] scanning {len(pgnFiles)} file(s) (cap={maxPgnFilesToScan}) for up to {maxPositions} positions...",
        flush=True,
    )

    openingTarget = int(maxPositions * 0.25)
    midTarget = int(maxPositions * 0.50)
    endTarget = maxPositions - openingTarget - midTarget

    openingCount = 0
    midCount = 0
    endCount = 0

    def tryAddBoard(b: chess.Board) -> bool:
        nonlocal duplicatesSkipped
        key = positionKey(b)
        if key in seenKeys:
            duplicatesSkipped += 1
            return False
        seenKeys.add(key)
        positions.append(b)
        return True

    for fileIndex, filePath in enumerate(pgnFiles, start=1):
        if len(positions) >= maxPositions:
            break

        print(
            f"[pgn] file {fileIndex}/{len(pgnFiles)}: {os.path.basename(filePath)}", flush=True)

        try:
            with open(filePath, "r", encoding="utf-8", errors="ignore") as fileHandle:
                gamesRead = 0
                while gamesRead < pgnGamesPerFileLimit and len(positions) < maxPositions:
                    game = chess.pgn.read_game(fileHandle)
                    if game is None:
                        break
                    gamesRead += 1

                    openingCand, midCand, endCand = collectGamePositionsByPhase(
                        game)

                    perGameCap = int(pgnPositionsPerGameLimit)
                    openCap = max(1, perGameCap // 4)
                    endCap = max(1, perGameCap // 4)
                    midCap = max(1, perGameCap - openCap - endCap)

                    random.shuffle(openingCand)
                    random.shuffle(midCand)
                    random.shuffle(endCand)

                    addedThisGame = 0

                    # Opening
                    if openingCount < openingTarget and openingCand:
                        takeGoal = min(openCap, openingTarget - openingCount,
                                       maxPositions - len(positions), len(openingCand))
                        for b in openingCand[:takeGoal]:
                            if len(positions) >= maxPositions or addedThisGame >= perGameCap:
                                break
                            if tryAddBoard(b):
                                openingCount += 1
                                addedThisGame += 1

                    # Endgame
                    if endCount < endTarget and endCand and addedThisGame < perGameCap and len(positions) < maxPositions:
                        remainingCap = perGameCap - addedThisGame
                        takeGoal = min(endCap, remainingCap, endTarget - endCount,
                                       maxPositions - len(positions), len(endCand))
                        for b in endCand[:takeGoal]:
                            if len(positions) >= maxPositions or addedThisGame >= perGameCap:
                                break
                            if tryAddBoard(b):
                                endCount += 1
                                addedThisGame += 1

                    # Middlegame
                    if midCount < midTarget and midCand and addedThisGame < perGameCap and len(positions) < maxPositions:
                        remainingCap = perGameCap - addedThisGame
                        takeGoal = min(midCap, remainingCap, midTarget - midCount,
                                       maxPositions - len(positions), len(midCand))
                        for b in midCand[:takeGoal]:
                            if len(positions) >= maxPositions or addedThisGame >= perGameCap:
                                break
                            if tryAddBoard(b):
                                midCount += 1
                                addedThisGame += 1

                    # Top-up
                    if len(positions) < maxPositions and addedThisGame < perGameCap:
                        remainingCap = perGameCap - addedThisGame
                        pool = openingCand + midCand + endCand
                        if pool:
                            random.shuffle(pool)
                            takeGoal = min(
                                remainingCap, maxPositions - len(positions), len(pool))
                            for b in pool[:takeGoal]:
                                if len(positions) >= maxPositions or addedThisGame >= perGameCap:
                                    break
                                if tryAddBoard(b):
                                    addedThisGame += 1

        except Exception as exc:
            print(f"[pgn] failed reading {filePath}: {exc}", flush=True)

    print(
        f"[pgn] done: collected {len(positions)} positions "
        f"(opening≈{openingCount}, mid≈{midCount}, end≈{endCount}, dupSkipped={duplicatesSkipped})",
        flush=True,
    )
    return positions


# -----------------------------
# Ridge regression utilities
# -----------------------------

def findBiasColumnIndex(X: np.ndarray, names: list[str]) -> tuple[Optional[int], str]:
    if X.shape[1] == 0:
        return None, "no features"

    lowered = [n.lower() for n in names]
    if "bias" in lowered:
        biasIndex = lowered.index("bias")
        if 0 <= biasIndex < X.shape[1]:
            if np.allclose(X[:, biasIndex], 1.0, atol=1e-12, rtol=0.0):
                return biasIndex, "used featureNames() 'bias' column"
            return None, "featureNames() has 'bias' but that column is not constant 1.0"

    for colIndex in range(X.shape[1]):
        if np.allclose(X[:, colIndex], 1.0, atol=1e-12, rtol=0.0):
            return colIndex, "used detected all-ones column"

    return None, "no constant-1.0 bias column found"


def trainModelRidgeClosedForm(xList: list[np.ndarray], yList: list[float], l2Lambda: float) -> EvalModel:
    X = np.vstack(xList).astype(np.float64)
    y = np.array(yList, dtype=np.float64)

    nSamples, nFeatures = X.shape

    onesCol = np.ones((nSamples, 1), dtype=np.float64)
    Xaug = np.hstack([X, onesCol])

    XtX = Xaug.T @ Xaug
    Xty = Xaug.T @ y

    regMatrix = np.eye(nFeatures + 1, dtype=np.float64) * float(l2Lambda)
    regMatrix[-1, -1] = 0.0

    A = XtX + regMatrix
    wAug = np.linalg.solve(A, Xty)

    learnedWeights = wAug[:-1].astype(np.float64)
    intercept = float(wAug[-1])

    names = featureNames()
    biasIndex, why = findBiasColumnIndex(X, names)
    if biasIndex is not None:
        learnedWeights = learnedWeights.copy()
        learnedWeights[biasIndex] += intercept
        intercept = 0.0
        print(
            f"[train][ridge] intercept folded into feature index {biasIndex} ({why})", flush=True)
    else:
        print(
            f"[train][ridge] WARNING: intercept={intercept:.6f} could not be folded ({why}); it will be ignored",
            flush=True,
        )

    print(
        f"[train][ridge] solved ridge regression. finalIntercept={intercept:.6f}", flush=True)
    return EvalModel(weights=learnedWeights)


# -----------------------------
# Fixed evaluation set
# -----------------------------

def evalSetPath() -> str:
    baseDir = os.path.dirname(weightsPath)
    return os.path.join(baseDir, "eval_set.json")


def loadEvalSetFens() -> Optional[list[str]]:
    path = evalSetPath()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fileHandle:
            obj = json.load(fileHandle)
        fens = obj.get("fens", None)
        if isinstance(fens, list) and all(isinstance(x, str) for x in fens) and len(fens) > 0:
            return fens
    except Exception:
        return None
    return None


def saveEvalSetFens(fens: list[str]) -> None:
    path = evalSetPath()
    obj = {"fens": fens}
    with open(path, "w", encoding="utf-8") as fileHandle:
        json.dump(obj, fileHandle, indent=2)
    print(
        f"[eval] saved fixed eval set to {path} (count={len(fens)})", flush=True)


def buildEvalSetFromPgnAndRandom(count: int) -> list[chess.Board]:
    savedState = random.getstate()
    random.seed(999123)

    pgnCount = int(count * 0.80)
    randCount = count - pgnCount

    pgnBoards = samplePgnPositions(pgnCount)
    boards: list[chess.Board] = []

    boards.extend(pgnBoards[:pgnCount])
    while len(boards) < pgnCount:
        boards.append(generateRandomPosition(randomPliesMax))

    for _ in range(randCount):
        boards.append(generateRandomPosition(randomPliesMax))

    random.setstate(savedState)
    return boards[:count]


def ensureEvalSet(fish: chess.engine.SimpleEngine, depth: int, evalCount: int) -> tuple[np.ndarray, np.ndarray]:
    fens = loadEvalSetFens()
    if fens is None:
        boards = buildEvalSetFromPgnAndRandom(evalCount)
        fens = [b.fen() for b in boards]
        saveEvalSetFens(fens)
    else:
        print(
            f"[eval] loaded fixed eval set from {evalSetPath()} (count={len(fens)})", flush=True)

    xList: list[np.ndarray] = []
    yRawList: list[float] = []

    total = min(evalCount, len(fens))
    with tqdm(total=total, desc="Processing eval set", unit="position", dynamic_ncols=True) as progressBar:
        for fen in fens[:total]:
            b = chess.Board(fen)
            yRaw = analyseBoardWithFish(fish, b, depth)
            xList.append(boardToFeatureVector(b))
            yRawList.append(float(yRaw))
            progressBar.update(1)

    Xeval = np.vstack(xList).astype(np.float64)
    yRawEval = np.array(yRawList, dtype=np.float64)
    return Xeval, yRawEval


# -----------------------------
# Metrics + squashing
# -----------------------------

def squashVectorTanh(yRaw: np.ndarray, yScale: float) -> np.ndarray:
    if yScale <= 0:
        return yRaw.astype(np.float64)
    return (float(yScale) * np.tanh(yRaw.astype(np.float64) / float(yScale))).astype(np.float64)


def computeRegressionMetrics(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    preds = X @ w
    mae = float(np.mean(np.abs(preds - y)))
    mse = float(np.mean((preds - y) ** 2))

    yStd = float(np.std(y))
    predStd = float(np.std(preds))

    yMin = float(np.min(y))
    yMax = float(np.max(y))
    pMin = float(np.min(preds))
    pMax = float(np.max(preds))

    if not np.isfinite(yStd) or not np.isfinite(predStd) or yStd < 1e-12 or predStd < 1e-12:
        corr = 0.0
    else:
        corrVal = np.corrcoef(preds, y)[0, 1]
        corr = float(corrVal) if np.isfinite(corrVal) else 0.0

    print(
        f"[metrics] y:  min={yMin:+.3f} max={yMax:+.3f} std={yStd:.6f} | "
        f"pred: min={pMin:+.3f} max={pMax:+.3f} std={predStd:.6f} | corr={corr:+.3f}",
        flush=True,
    )
    return mae, mse, corr


# -----------------------------
# Near-dupe reduction (feature buckets)
# -----------------------------

def featureBucketKey(featureVector: np.ndarray, decimals: int = 2) -> str:
    """
    Make a coarse hash of the feature vector to treat "very similar" positions as near-duplicates.

    How it works:
      - rounds each feature to a fixed number of decimals
      - converts to a compact string

    Why it helps:
      - You can have many positions that differ by move counters / trivial rearrangements,
        still "same-ish" in features.
      - Capping per bucket makes training more diverse.
    """
    rounded = np.round(featureVector.astype(np.float64), decimals=decimals)
    # Turning into bytes/string is fast enough at your scale; this is simple + reliable.
    return rounded.tobytes().hex()


# -----------------------------
# Main training entry point
# -----------------------------

def trainEvalModel(
    stockfishPath: str,
    sampleCount: int,
    stockfishDepth: int,
    seed: int = 1234,
    shouldStop: Optional[Callable[[], bool]] = None,
    reduceDupes: bool = True,
    maxPerFeatureBucket: int = 6,
    featureBucketDecimals: int = 2,
) -> EvalModel:
    """
    Collect samples, label with Stockfish, fit linear model.

    New arguments:
      - shouldStop: callback; if returns True we stop collecting early
      - reduceDupes: enables PGN pool + near-dupe buckets
      - maxPerFeatureBucket: cap how many samples we keep from the same feature bucket
      - featureBucketDecimals: rounding for bucket creation (higher => stricter similarity)
    """
    random.seed(seed)
    np.random.seed(seed)

    names = featureNames()
    featureCount = len(names)

    loadedModel = loadModel(weightsPath)
    if loadedModel is None:
        loadedModel = EvalModel(weights=np.zeros(
            featureCount, dtype=np.float64))
        print("[train] no existing weights.json; starting from zeros", flush=True)
    else:
        if len(loadedModel.weights) != featureCount:
            print(
                f"[train] WARNING: weights.json has {len(loadedModel.weights)} weights but feature vector has {featureCount}. "
                "Resetting weights to zeros to avoid mismatch.",
                flush=True,
            )
            loadedModel = EvalModel(weights=np.zeros(
                featureCount, dtype=np.float64))
        print("[train] loaded existing weights.json; continuing training", flush=True)

    model = loadedModel
    selfPlayEngine = AlphaBetaEngine(model)

    weightsAbsSum = float(np.sum(np.abs(model.weights)))
    useSelfPlay = weightsAbsSum > 1e-6
    print(
        f"[train] starting: samples={sampleCount}, depth={stockfishDepth}", flush=True)
    print(
        f"[train] weights abs sum: {weightsAbsSum:.6f}. Self-play enabled: {useSelfPlay}", flush=True)

    evalCount = 2000

    # Pre-sample PGN boards
    pgnTarget = int(sampleCount * trainMixPgn)
    pgnBoards = samplePgnPositions(pgnTarget)

    # --- Dupe reduction: use a shuffled PGN pool iterator instead of random.choice ---
    pgnPool: list[chess.Board] = list(pgnBoards)
    if reduceDupes and pgnPool:
        random.shuffle(pgnPool)
    pgnPoolIndex = 0

    xList: list[np.ndarray] = []
    yRawTrainList: list[float] = []

    seenTrainKeys: set[str] = set()
    duplicatesSkipped = 0
    attempts = 0

    # --- Near-dupe buckets ---
    bucketCounts: dict[str, int] = {}

    with chess.engine.SimpleEngine.popen_uci(stockfishPath) as fish:
        Xeval, yRawEval = ensureEvalSet(
            fish=fish, depth=stockfishDepth, evalCount=evalCount)

        selfPlayCount = 0
        randomCount = 0
        pgnCount = 0
        nearDupesSkipped = 0

        progressBar = tqdm(
            total=sampleCount, desc="Collecting samples", unit="sample", dynamic_ncols=True)

        try:
            # allow some waste but avoid infinite loops
            maxAttempts = int(sampleCount * 10)

            while len(xList) < sampleCount and attempts < maxAttempts:
                attempts += 1

                if shouldStop is not None and shouldStop():
                    print(
                        f"[train] early stop requested: using {len(xList)} collected sample(s)", flush=True)
                    break

                r = random.random()
                selfPlayMix = trainMixSelfPlay if useSelfPlay else 0.0

                if useSelfPlay and r < selfPlayMix:
                    board = generateSelfPlayPosition(
                        selfPlayEngine, selfPlayMaxPlies, selfPlayDepth, selfPlayRandomness)
                    sourceTag = "self"

                elif r < selfPlayMix + trainMixPgn and pgnPool:
                    if reduceDupes:
                        board = pgnPool[pgnPoolIndex].copy(stack=False)
                        pgnPoolIndex += 1
                        if pgnPoolIndex >= len(pgnPool):
                            random.shuffle(pgnPool)
                            pgnPoolIndex = 0
                    else:
                        board = random.choice(pgnPool).copy(stack=False)
                    sourceTag = "pgn"

                else:
                    board = generateRandomPosition(randomPliesMax)
                    sourceTag = "rand"

                # Exact dedupe
                key = positionKey(board)
                if key in seenTrainKeys:
                    duplicatesSkipped += 1
                    progressBar.set_postfix({
                        "self": selfPlayCount, "pgn": pgnCount, "rand": randomCount,
                        "dup": duplicatesSkipped, "nearDup": nearDupesSkipped
                    })
                    continue
                seenTrainKeys.add(key)

                # Build feature vector (we need this for near-dupe bucketing)
                x = boardToFeatureVector(board)
                if len(x) != featureCount:
                    raise RuntimeError(
                        f"boardToFeatureVector returned length {len(x)} but featureNames() length is {featureCount}. These must match."
                    )

                # Near-dupe reduction by bucket cap
                if reduceDupes and maxPerFeatureBucket > 0:
                    bucketKey = featureBucketKey(
                        x, decimals=featureBucketDecimals)
                    currentCount = bucketCounts.get(bucketKey, 0)
                    if currentCount >= maxPerFeatureBucket:
                        nearDupesSkipped += 1
                        progressBar.set_postfix({
                            "self": selfPlayCount, "pgn": pgnCount, "rand": randomCount,
                            "dup": duplicatesSkipped, "nearDup": nearDupesSkipped
                        })
                        continue
                    bucketCounts[bucketKey] = currentCount + 1

                # Label AFTER passing dedupe filters (don’t waste Stockfish time)
                yRaw = float(analyseBoardWithFish(fish, board, stockfishDepth))
                yRawTrainList.append(yRaw)
                xList.append(x)

                if sourceTag == "self":
                    selfPlayCount += 1
                elif sourceTag == "pgn":
                    pgnCount += 1
                else:
                    randomCount += 1

                progressBar.update(1)
                progressBar.set_postfix({
                    "self": selfPlayCount, "pgn": pgnCount, "rand": randomCount,
                    "dup": duplicatesSkipped, "nearDup": nearDupesSkipped
                })

        finally:
            progressBar.close()

        collectedCount = len(xList)
        if collectedCount < 50:
            raise RuntimeError(
                f"Too few samples collected ({collectedCount}). Increase Samples or don't stop so early."
            )

        if collectedCount < sampleCount:
            print(
                f"[train] NOTE: collected {collectedCount}/{sampleCount} samples "
                f"(attempts={attempts}, dupSkipped={duplicatesSkipped}, nearDupSkipped={nearDupesSkipped}).",
                flush=True,
            )
            sampleCount = collectedCount

        print(
            f"[train] sample collection done. Self-play={selfPlayCount}, PGN={pgnCount}, Random={randomCount}, "
            f"dupSkipped={duplicatesSkipped}, nearDupSkipped={nearDupesSkipped}, attempts={attempts}",
            flush=True,
        )

        # -----------------------------
        # Candidate sweep: pick yScale + L2 that generalizes best
        # -----------------------------
        yRawTrain = np.array(yRawTrainList, dtype=np.float64)

        candidateYScales = [4.0, 6.0, 10.0, 15.0]
        candidateL2 = [0.5, 1.0, 3.0, 10.0]

        bestModel: EvalModel | None = None
        bestSummary = None
        bestScore = -1e9

        Xtrain = np.vstack(xList).astype(np.float64)

        for candScale in candidateYScales:
            yTrain = squashVectorTanh(yRawTrain, candScale)
            yEval = squashVectorTanh(yRawEval, candScale)

            for candL2 in candidateL2:
                trainedCand = trainModelRidgeClosedForm(
                    xList=xList, yList=yTrain.tolist(), l2Lambda=float(candL2)
                )

                trainMae, trainMse, trainCorr = computeRegressionMetrics(
                    Xtrain, yTrain, trainedCand.weights)
                evalMae, evalMse, evalCorr = computeRegressionMetrics(
                    Xeval, yEval, trainedCand.weights)

                score = float(evalCorr) - 0.06 * float(evalMae)

                print(
                    f"[sweep] yScale={candScale:.1f} l2={candL2:.2f} | "
                    f"eval: corr={evalCorr:+.3f} mae={evalMae:.3f} mse={evalMse:.4f} | "
                    f"train: corr={trainCorr:+.3f} mae={trainMae:.3f} mse={trainMse:.4f} | "
                    f"score={score:+.3f}",
                    flush=True
                )

                if score > bestScore:
                    bestScore = score
                    bestModel = trainedCand
                    bestSummary = (
                        candScale, candL2,
                        evalCorr, evalMae, evalMse,
                        trainCorr, trainMae, trainMse
                    )

        if bestModel is None or bestSummary is None:
            raise RuntimeError("candidate sweep failed to produce a model")

        (
            candScale, candL2,
            bestEvalCorr, bestEvalMae, bestEvalMse,
            bestTrainCorr, bestTrainMae, bestTrainMse
        ) = bestSummary

        print(
            f"[pick] BEST: yScale={candScale:.1f} l2={candL2:.2f} | "
            f"eval corr={bestEvalCorr:+.3f} mae={bestEvalMae:.3f} mse={bestEvalMse:.4f} | "
            f"train corr={bestTrainCorr:+.3f} mae={bestTrainMae:.3f} mse={bestTrainMse:.4f}",
            flush=True
        )

        trainedModel = bestModel
        trainedModel.labelScale = float(candScale)

    saveModel(trainedModel, weightsPath)
    print(f"[train] saved weights to {weightsPath}", flush=True)
    return trainedModel
