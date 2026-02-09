"""Training utilities for fitting the evaluation model to Stockfish."""

import os
import random

import chess
import chess.engine
import chess.pgn
import numpy as np
from tqdm import tqdm

from config import (labelTimeMs, labelUseTimeMs, maxPgnFilesToScan, pgnDir,
                    pgnGamesPerFileLimit, pgnPositionsPerGameLimit,
                    randomPliesMax, selfPlayDepth, selfPlayMaxPlies,
                    selfPlayRandomness, trainMixPgn, trainMixRandom,
                    trainMixSelfPlay, weightsPath)
from engine import AlphaBetaEngine
from features import boardToFeatureVector, featureNames
from model import EvalModel, loadModel, saveModel


def stockfishScoreToPawns(score: chess.engine.PovScore) -> float:
    """Convert a Stockfish POV score to a pawn-scale float."""
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


def generateRandomPosition(maxPlies: int) -> chess.Board:
    """Generate a random legal position by playing random plies."""
    board = chess.Board()
    plies = random.randint(2, maxPlies)
    for _ in range(plies):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board


def generateSelfPlayPosition(engine: AlphaBetaEngine, maxPlies: int, depth: int, randomness: float) -> chess.Board:
    """Generate a position via self-play with optional random moves."""
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


def iterPgnFiles() -> list[str]:
    """List PGN files available for sampling training positions."""
    if not os.path.isdir(pgnDir):
        return []
    files = [os.path.join(pgnDir, f) for f in os.listdir(
        pgnDir) if f.lower().endswith(".pgn")]
    print(f"[train] found {len(files)} PGN files in {pgnDir}", flush=True)
    return files


def samplePgnPositions(maxPositions: int) -> list[chess.Board]:
    """Sample mid-game positions from PGN files for training labels."""
    positions: list[chess.Board] = []
    pgnFiles = iterPgnFiles()
    if not pgnFiles:
        print("[pgn] no PGN files found", flush=True)
        return positions
    random.shuffle(pgnFiles)
    pgnFiles = pgnFiles[:maxPgnFilesToScan]
    print(
        f"[pgn] scanning {len(pgnFiles)} file(s) (cap={maxPgnFilesToScan}) for up to {maxPositions} positions...",
        flush=True
    )
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
                    board = game.board()
                    ply = 0
                    addedFromThisGame = 0
                    for move in game.mainline_moves():
                        board.push(move)
                        ply += 1
                        if ply >= 16 and ply % 4 == 0 and not board.is_game_over():
                            positions.append(board.copy(stack=False))
                            addedFromThisGame += 1
                            if addedFromThisGame >= pgnPositionsPerGameLimit:
                                break
                            if len(positions) >= maxPositions:
                                break
        except Exception as exc:
            print(f"[pgn] failed reading {filePath}: {exc}", flush=True)
    print(f"[pgn] done: collected {len(positions)} positions", flush=True)
    return positions


def trainModelSgd(
    initialModel: EvalModel,
    xList: list[np.ndarray],
    yList: list[float],
    learningRate: float,
    epochs: int,
    l2Lambda: float,
    batchSize: int,
    seed: int,
) -> EvalModel:
    """Train a linear model with mini-batch SGD and L2 regularization."""
    rng = np.random.default_rng(seed)
    X = np.vstack(xList).astype(np.float64)
    y = np.array(yList, dtype=np.float64)
    weights = initialModel.weights.astype(np.float64).copy()
    n = X.shape[0]
    indices = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(indices)
        epochMse = 0.0
        batchCount = 0
        for start in range(0, n, batchSize):
            batchIdx = indices[start:start + batchSize]
            Xb = X[batchIdx]
            yb = y[batchIdx]
            preds = Xb @ weights
            errors = preds - yb
            grad = (2.0 / len(batchIdx)) * (Xb.T @ errors)
            grad += 2.0 * l2Lambda * weights
            weights -= learningRate * grad
            batchMse = float(np.mean(errors * errors))
            epochMse += batchMse
            batchCount += 1
        epochMse /= max(1, batchCount)
        print(
            f"[train] epoch {epoch+1}/{epochs}: mse={epochMse:.6f}", flush=True)
    return EvalModel(weights=weights)


def trainEvalModel(stockfishPath: str, sampleCount: int, stockfishDepth: int, seed: int = 1234) -> EvalModel:
    """Collect samples, label with Stockfish, and fit a new model."""
    random.seed(seed)
    np.random.seed(seed)
    loadedModel = loadModel(weightsPath)
    if loadedModel is None:
        initialWeights = np.zeros(len(featureNames()), dtype=np.float64)
        loadedModel = EvalModel(weights=initialWeights)
        print("[train] no existing weights.json; starting from zeros", flush=True)
    else:
        print("[train] loaded existing weights.json; continuing training", flush=True)
    model = loadedModel
    selfPlayEngine = AlphaBetaEngine(model)
    weightsAbsSum = float(np.sum(np.abs(model.weights)))
    useSelfPlay = weightsAbsSum > 1e-6
    print(
        f"[train] weights abs sum: {weightsAbsSum:.6f}. Self-play enabled: {useSelfPlay}", flush=True)
    pgnTarget = int(sampleCount * trainMixPgn)
    pgnBoards = samplePgnPositions(pgnTarget)
    xList: list[np.ndarray] = []
    yList: list[float] = []
    with chess.engine.SimpleEngine.popen_uci(stockfishPath) as fish:
        selfPlayCount = 0
        randomCount = 0
        pgnCount = 0
        progressBar = tqdm(
            total=sampleCount, desc="Collecting samples", unit="sample", dynamic_ncols=True)
        try:
            for i in range(sampleCount):
                r = random.random()
                selfPlayMix = trainMixSelfPlay if useSelfPlay else 0.0
                if useSelfPlay and r < selfPlayMix:
                    board = generateSelfPlayPosition(
                        selfPlayEngine, selfPlayMaxPlies, selfPlayDepth, selfPlayRandomness)
                    selfPlayCount += 1
                elif r < selfPlayMix + trainMixPgn and pgnBoards:
                    board = random.choice(pgnBoards).copy(stack=False)
                    pgnCount += 1
                else:
                    board = generateRandomPosition(randomPliesMax)
                    randomCount += 1
                if labelUseTimeMs and labelTimeMs > 0:
                    info = fish.analyse(board, chess.engine.Limit(
                        time=labelTimeMs / 1000.0))
                else:
                    info = fish.analyse(
                        board, chess.engine.Limit(depth=stockfishDepth))
                y = stockfishScoreToPawns(info["score"])
                x = boardToFeatureVector(board)
                xList.append(x)
                yList.append(y)
                if (i + 1) % 4 == 0 or (i + 1) == sampleCount:
                    progressBar.update(1)
                    progressBar.set_postfix({
                        "self": selfPlayCount,
                        "pgn": pgnCount,
                        "rand": randomCount
                    })
        finally:
            progressBar.close()
    print(
        f"[train] sample collection done. Self-play={selfPlayCount}, PGN={pgnCount}, Random={randomCount}",
        flush=True
    )
    learningRate = 0.01
    epochs = 3
    l2Lambda = 1e-4
    batchSize = 256
    trainedModel = trainModelSgd(
        initialModel=model,
        xList=xList,
        yList=yList,
        learningRate=learningRate,
        epochs=epochs,
        l2Lambda=l2Lambda,
        batchSize=batchSize,
        seed=seed,
    )
    X = np.vstack(xList)
    yArr = np.array(yList, dtype=np.float64)
    preds = X @ trainedModel.weights
    mae = float(np.mean(np.abs(preds - yArr)))
    print(f"[train] done. Training MAE ≈ {mae:.3f} pawns", flush=True)
    saveModel(trainedModel, weightsPath)
    print(f"[train] saved weights to {weightsPath}", flush=True)
    return trainedModel
