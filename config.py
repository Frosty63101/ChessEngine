"""Centralized configuration values for the chess engine and trainer."""

import os

stockfishPath = r"D:\ChessEngine\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

# Data paths.
imageDir = os.path.join(os.path.dirname(__file__), "images")
pgnDir = os.path.join(os.path.dirname(__file__), "pgns_small")

# PGN sampling limits.
maxPgnFilesToScan = 100
pgnGamesPerFileLimit = 25
pgnPositionsPerGameLimit = 8

weightsPath = os.path.join(os.path.dirname(__file__), "weights.json")

# Training and evaluation defaults.
defaultStockfishDepth = 12

defaultTrainSamples = 5000
trainMixSelfPlay = 0.1
trainMixPgn = 0.65
trainMixRandom = 0.25

selfPlayDepth = 3
selfPlayRandomness = 0.1
selfPlayMaxPlies = 30

randomPliesMax = 14

labelUseTimeMs = 0
labelTimeMs = 250

defaultSearchDepth = 6
defaultThinkTimeMs = 350

qMaxDepth = 8

guiStockfishDepth = 12
guiStockfishTimeMs = 0


# -----------------------------
# Adaptive training
# -----------------------------
adaptiveEnabled = True
adaptiveValFraction = 0.12

baseLearningRate = 0.001
baseEpochs = 3
baseL2Lambda = 1e-3
baseBatchSize = 256

adaptiveMinDelta = 0.01  # pawns
adaptivePatience = 1

adaptiveLrDecay = 0.55
adaptiveL2Grow = 1.8

adaptiveMinLearningRate = 1e-5
adaptiveMaxLearningRate = 0.05

adaptiveMinL2Lambda = 0.0
adaptiveMaxL2Lambda = 5e-2

adaptiveAllowExtraEpochs = True
adaptiveMaxExtraEpochs = 5

adaptiveEarlyStopNoImproveEpochs = 4

# If validation MAE gets worse by "worsenDelta" for "patience" epochs,
# rollback weights to the best seen so far and dampen learning.
adaptiveRollbackEnabled = True

# How much worse (in pawns MAE) counts as "we drifted away" from the best.
adaptiveRollbackWorsenDelta = 0.015

# How many consecutive "worse-than-best+delta" epochs before rollback.
adaptiveRollbackPatience = 1

# After rollback, apply an extra LR decay (in addition to normal tuning).
adaptiveRollbackExtraLrDecay = 0.5

# After rollback, optionally increase L2 a bit to stabilize.
adaptiveRollbackExtraL2Grow = 1.25
