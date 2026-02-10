"""Training utilities for fitting the evaluation model to Stockfish."""

import os
import random
from dataclasses import dataclass

import chess
import chess.engine
import chess.pgn
import numpy as np
from tqdm import tqdm

from config import (  # Adaptive settings; Rollback settings
    adaptiveAllowExtraEpochs, adaptiveEarlyStopNoImproveEpochs,
    adaptiveEnabled, adaptiveL2Grow, adaptiveLrDecay, adaptiveMaxExtraEpochs,
    adaptiveMaxL2Lambda, adaptiveMaxLearningRate, adaptiveMinDelta,
    adaptiveMinL2Lambda, adaptiveMinLearningRate, adaptivePatience,
    adaptiveRollbackEnabled, adaptiveRollbackExtraL2Grow,
    adaptiveRollbackExtraLrDecay, adaptiveRollbackPatience,
    adaptiveRollbackWorsenDelta, adaptiveValFraction, baseBatchSize,
    baseEpochs, baseL2Lambda, baseLearningRate, labelTimeMs, labelUseTimeMs,
    maxPgnFilesToScan, pgnDir, pgnGamesPerFileLimit, pgnPositionsPerGameLimit,
    randomPliesMax, selfPlayDepth, selfPlayMaxPlies, selfPlayRandomness,
    trainMixPgn, trainMixRandom, trainMixSelfPlay, weightsPath)
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
        flush=True,
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


@dataclass
class EpochMetrics:
    trainMae: float
    valMae: float
    trainMse: float
    valMse: float
    corr: float


class AdaptiveTuner:
    """
    Rule-based tuner:
      - If validation improves => keep settings
      - If stalls => decay LR, maybe increase L2
      - Supports rollback-to-best when validation clearly drifts
    """

    def __init__(
        self,
        learningRate: float,
        l2Lambda: float,
        batchSize: int,
        minDelta: float,
        patience: int,
        lrDecay: float,
        l2Grow: float,
        minLearningRate: float,
        maxLearningRate: float,
        minL2Lambda: float,
        maxL2Lambda: float,
        earlyStopNoImproveEpochs: int,
    ):
        self.learningRate = float(learningRate)
        self.l2Lambda = float(l2Lambda)
        self.batchSize = int(batchSize)

        self.minDelta = float(minDelta)
        self.patience = int(patience)
        self.lrDecay = float(lrDecay)
        self.l2Grow = float(l2Grow)

        self.minLearningRate = float(minLearningRate)
        self.maxLearningRate = float(maxLearningRate)
        self.minL2Lambda = float(minL2Lambda)
        self.maxL2Lambda = float(maxL2Lambda)

        self.earlyStopNoImproveEpochs = int(earlyStopNoImproveEpochs)

        self.bestValMae = float("inf")
        self.noImproveStreak = 0
        self.totalNoImprove = 0

    def _clamp(self) -> None:
        self.learningRate = max(self.minLearningRate, min(
            self.maxLearningRate, self.learningRate))
        self.l2Lambda = max(self.minL2Lambda, min(
            self.maxL2Lambda, self.l2Lambda))

    def applyRollbackDampening(self) -> str:
        """
        After rollback, dampen the step size and add a bit more regularization
        so you don't immediately walk away from the good basin again.
        """
        oldLr = self.learningRate
        oldL2 = self.l2Lambda

        self.learningRate *= float(adaptiveRollbackExtraLrDecay)
        self.l2Lambda *= float(adaptiveRollbackExtraL2Grow)
        self._clamp()

        return f"rollback dampen: lr {oldLr:.6g}→{self.learningRate:.6g} | l2 {oldL2:.6g}→{self.l2Lambda:.6g}"

    def updateForStall(self, metrics: EpochMetrics) -> tuple[bool, str]:
        """
        Called when validation did not improve.
        Returns (shouldStop, message)
        """
        self.noImproveStreak += 1
        self.totalNoImprove += 1

        actionParts: list[str] = []

        if self.noImproveStreak >= self.patience:
            oldLr = self.learningRate
            self.learningRate *= self.lrDecay
            self._clamp()
            if self.learningRate != oldLr:
                actionParts.append(f"lr {oldLr:.6g}→{self.learningRate:.6g}")

            # Overfitting/noise heuristic: train MAE much smaller than val MAE
            if metrics.trainMae + 1e-9 < metrics.valMae * 0.85:
                oldL2 = self.l2Lambda
                self.l2Lambda *= self.l2Grow
                self._clamp()
                if self.l2Lambda != oldL2:
                    actionParts.append(f"l2 {oldL2:.6g}→{self.l2Lambda:.6g}")

            self.noImproveStreak = 0  # reset after taking an action

        shouldStop = self.totalNoImprove >= self.earlyStopNoImproveEpochs
        if not actionParts:
            actionParts.append("no change")

        return shouldStop, " | ".join(actionParts)

    def updateBest(self, valMae: float) -> bool:
        """
        Update best validation score. Returns True if improved enough.
        """
        improved = valMae < (self.bestValMae - self.minDelta)
        if improved:
            self.bestValMae = valMae
            self.noImproveStreak = 0
        return improved


def _computeMetrics(weights: np.ndarray, Xtrain: np.ndarray, yTrain: np.ndarray, Xval: np.ndarray, yVal: np.ndarray) -> EpochMetrics:
    trainPred = Xtrain @ weights
    valPred = Xval @ weights

    trainErr = trainPred - yTrain
    valErr = valPred - yVal

    trainMae = float(np.mean(np.abs(trainErr)))
    valMae = float(np.mean(np.abs(valErr)))

    trainMse = float(np.mean(trainErr * trainErr))
    valMse = float(np.mean(valErr * valErr))

    valStdPred = float(np.std(valPred))
    valStdY = float(np.std(yVal))
    if valStdPred < 1e-12 or valStdY < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(valPred, yVal)[0, 1])

    return EpochMetrics(trainMae=trainMae, valMae=valMae, trainMse=trainMse, valMse=valMse, corr=corr)


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
    """Train a linear model with mini-batch SGD and L2 regularization (non-adaptive)."""
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
            epochMse += float(np.mean(errors * errors))
            batchCount += 1
        epochMse /= max(1, batchCount)
        print(
            f"[train] epoch {epoch+1}/{epochs}: mse={epochMse:.6f}", flush=True)

    return EvalModel(weights=weights)


def trainModelSgdAdaptive(
    initialModel: EvalModel,
    xList: list[np.ndarray],
    yList: list[float],
    learningRate: float,
    epochs: int,
    l2Lambda: float,
    batchSize: int,
    seed: int,
    valFraction: float,
) -> EvalModel:
    """
    Adaptive training with:
      - train/val split
      - best checkpoint tracking
      - rollback-to-best when validation drifts
      - LR/L2 tuning on stalls
    """
    rng = np.random.default_rng(seed)

    Xall = np.vstack(xList).astype(np.float64)
    yAll = np.array(yList, dtype=np.float64)
    n = Xall.shape[0]

    if n < 50:
        print(
            "[train][adaptive] too few samples; falling back to non-adaptive SGD", flush=True)
        return trainModelSgd(initialModel, xList, yList, learningRate, epochs, l2Lambda, batchSize, seed)

    indices = np.arange(n)
    rng.shuffle(indices)

    valCount = int(max(1, min(n - 1, round(n * float(valFraction)))))
    valIdx = indices[:valCount]
    trainIdx = indices[valCount:]

    Xval = Xall[valIdx]
    yVal = yAll[valIdx]
    Xtrain = Xall[trainIdx]
    yTrain = yAll[trainIdx]

    weights = initialModel.weights.astype(np.float64).copy()
    trainN = Xtrain.shape[0]
    trainIndices = np.arange(trainN)

    tuner = AdaptiveTuner(
        learningRate=learningRate,
        l2Lambda=l2Lambda,
        batchSize=batchSize,
        minDelta=adaptiveMinDelta,
        patience=adaptivePatience,
        lrDecay=adaptiveLrDecay,
        l2Grow=adaptiveL2Grow,
        minLearningRate=adaptiveMinLearningRate,
        maxLearningRate=adaptiveMaxLearningRate,
        minL2Lambda=adaptiveMinL2Lambda,
        maxL2Lambda=adaptiveMaxL2Lambda,
        earlyStopNoImproveEpochs=adaptiveEarlyStopNoImproveEpochs,
    )

    maxEpochs = int(epochs) + (int(adaptiveMaxExtraEpochs)
                               if adaptiveAllowExtraEpochs else 0)

    bestWeights = weights.copy()
    bestValMae = float("inf")

    driftStreak = 0  # counts consecutive "worse-than-best+delta" epochs

    for epoch in range(maxEpochs):
        rng.shuffle(trainIndices)

        lrNow = tuner.learningRate
        l2Now = tuner.l2Lambda
        bsNow = tuner.batchSize

        # One epoch of SGD on train split
        for start in range(0, trainN, bsNow):
            batchIdx = trainIndices[start:start + bsNow]
            Xb = Xtrain[batchIdx]
            yb = yTrain[batchIdx]

            preds = Xb @ weights
            errors = preds - yb
            grad = (2.0 / len(batchIdx)) * (Xb.T @ errors)
            grad += 2.0 * l2Now * weights
            weights -= lrNow * grad

        metrics = _computeMetrics(weights, Xtrain, yTrain, Xval, yVal)

        # Update best checkpoint if improved enough
        improved = tuner.updateBest(metrics.valMae)
        if improved:
            bestValMae = metrics.valMae
            bestWeights = weights.copy()
            driftStreak = 0

        # Drift detection: val got meaningfully worse than best
        drifted = metrics.valMae > (
            bestValMae + float(adaptiveRollbackWorsenDelta))
        if adaptiveRollbackEnabled and drifted:
            driftStreak += 1
        else:
            driftStreak = 0

        rollbackMsg = ""
        if adaptiveRollbackEnabled and driftStreak >= int(adaptiveRollbackPatience):
            # Rollback weights to best and dampen hyperparams
            weights = bestWeights.copy()
            rollbackMsg = tuner.applyRollbackDampening()
            driftStreak = 0  # reset after rollback

            # Recompute metrics post-rollback for logging clarity
            metrics = _computeMetrics(weights, Xtrain, yTrain, Xval, yVal)

        # If not improved, do stall-based tuning and maybe early stop
        if not improved:
            shouldStop, actionMsg = tuner.updateForStall(metrics)
        else:
            shouldStop, actionMsg = (False, "val improved")

        extra = f" | {rollbackMsg}" if rollbackMsg else ""
        print(
            "[train][adaptive] "
            f"epoch {epoch+1}/{maxEpochs} | "
            f"trainMae={metrics.trainMae:.3f} valMae={metrics.valMae:.3f} | "
            f"bestValMae={bestValMae:.3f} | "
            f"trainMse={metrics.trainMse:.4f} valMse={metrics.valMse:.4f} | "
            f"corr={metrics.corr:+.3f} | "
            f"lr={tuner.learningRate:.6g} l2={tuner.l2Lambda:.6g} bs={tuner.batchSize} | "
            f"action={actionMsg}{extra}",
            flush=True,
        )

        # Only early-stop after you've at least run your base epoch budget
        if (epoch + 1) >= int(epochs) and shouldStop:
            print("[train][adaptive] early stop: validation not improving", flush=True)
            break

    # Return the best checkpoint on validation
    return EvalModel(weights=bestWeights)


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

                progressBar.update(1)
                progressBar.set_postfix(
                    {"self": selfPlayCount, "pgn": pgnCount, "rand": randomCount})
        finally:
            progressBar.close()

    print(
        f"[train] sample collection done. Self-play={selfPlayCount}, PGN={pgnCount}, Random={randomCount}",
        flush=True,
    )

    learningRate = baseLearningRate
    epochs = baseEpochs
    l2Lambda = baseL2Lambda
    batchSize = baseBatchSize

    if adaptiveEnabled:
        print(
            f"[train][adaptive] enabled: valFraction={adaptiveValFraction} "
            f"(base lr={learningRate}, epochs={epochs}, l2={l2Lambda}, bs={batchSize})",
            flush=True,
        )
        trainedModel = trainModelSgdAdaptive(
            initialModel=model,
            xList=xList,
            yList=yList,
            learningRate=learningRate,
            epochs=epochs,
            l2Lambda=l2Lambda,
            batchSize=batchSize,
            seed=seed,
            valFraction=adaptiveValFraction,
        )
    else:
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

    # Report training MAE (not validation — just a basic sanity number)
    X = np.vstack(xList).astype(np.float64)
    yArr = np.array(yList, dtype=np.float64)
    preds = X @ trainedModel.weights
    mae = float(np.mean(np.abs(preds - yArr)))
    print(f"[train] done. Training MAE ≈ {mae:.3f} pawns", flush=True)

    saveModel(trainedModel, weightsPath)
    print(f"[train] saved weights to {weightsPath}", flush=True)
    return trainedModel
