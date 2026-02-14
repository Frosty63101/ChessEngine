"""Centralized configuration values for the chess engine and trainer."""

import os

# -----------------------------
# Paths
# -----------------------------
stockfishPath = r"D:\ChessEngine\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

projectDir = os.path.dirname(__file__)

imageDir = os.path.join(projectDir, "images")
pgnDir = os.path.join(projectDir, "pgns_small")

weightsPath = os.path.join(projectDir, "weights.json")
modelEloPath = os.path.join(projectDir, "model_elo.json")

# -----------------------------
# Training data sampling
# -----------------------------
# You have 15k+ PGNs. Scanning more per run improves coverage but costs time.
# With your phase-aware sampling, scanning 250-600 files is usually a big step up from 100.
maxPgnFilesToScan = 300

# Per file: how many games to parse. 25 is okay; 40 gives more variety per file.
pgnGamesPerFileLimit = 35

# Per game: how many positions to extract max. You already phase-split; 8 is ok.
# If you want more endgame/phase variety, 10-12 helps.
pgnPositionsPerGameLimit = 10

# -----------------------------
# Stockfish labeling (teacher)
# -----------------------------
defaultStockfishDepth = 12

# If labelUseTimeMs=1, use labelTimeMs instead of depth.
# Time mode is more stable across positions; depth mode is more deterministic.
labelUseTimeMs = 0
labelTimeMs = 250

# -----------------------------
# Train mixture
# -----------------------------
# With dedupe enabled, you can lean heavier on PGN safely.
# Random positions are often garbage/unrealistic after ~10 plies; keep them but not too high.
defaultTrainSamples = 8000

trainMixSelfPlay = 0.05
trainMixPgn = 0.80
trainMixRandom = 0.15

# -----------------------------
# Self-play generation
# -----------------------------
# Self-play depth is low; keep it cheap. Randomness prevents mode collapse.
selfPlayDepth = 3
selfPlayRandomness = 0.12
selfPlayMaxPlies = 34

# Random plies: 14 is okay. Higher makes more nonsense positions.
randomPliesMax = 14

# -----------------------------
# Ridge regression strength
# -----------------------------
# This matters a LOT now that you’re using closed-form ridge.
# With richer features, a slightly stronger L2 usually generalizes better.
# Try 3e-3 to 1e-2 if eval MAE is jumpy.
baseL2Lambda = 0.004

# -----------------------------
# Engine defaults
# -----------------------------
defaultSearchDepth = 6
defaultThinkTimeMs = 350

qMaxDepth = 8

guiStockfishDepth = 12
guiStockfishTimeMs = 0

# -----------------------------
# Adaptive training
# -----------------------------
# NOTE:
# Your current trainer is closed-form ridge (one-shot solve), so LR/epochs/batch/adaptive
# don't apply unless you still have an older SGD/epoch-based path somewhere.
# Leaving these here won’t hurt, but they are unused in the ridge trainer.
adaptiveEnabled = False
adaptiveValFraction = 0.12

baseLearningRate = 0.001
baseEpochs = 3
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

adaptiveRollbackEnabled = True
adaptiveRollbackWorsenDelta = 0.015
adaptiveRollbackPatience = 1
adaptiveRollbackExtraLrDecay = 0.5
adaptiveRollbackExtraL2Grow = 1.25
