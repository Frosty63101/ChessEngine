"""Centralized configuration values for the chess engine and trainer."""

import os

stockfishPath = r"D:\ChessEngine\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

# Data paths.
imageDir = os.path.join(os.path.dirname(__file__), "images")
pgnDir = os.path.join(os.path.dirname(__file__), "pgns_small")

# PGN sampling limits.
maxPgnFilesToScan = 100
pgnGamesPerFileLimit = 25
pgnPositionsPerGameLimit = 6

weightsPath = os.path.join(os.path.dirname(__file__), "weights.json")

# Training and evaluation defaults.
defaultStockfishDepth = 10

defaultTrainSamples = 5000
trainMixSelfPlay = 0.03
trainMixPgn = 0.65
trainMixRandom = 0.32

selfPlayDepth = 3
selfPlayRandomness = 0.15
selfPlayMaxPlies = 30

randomPliesMax = 14

labelUseTimeMs = 0
labelTimeMs = 150

defaultSearchDepth = 4
defaultThinkTimeMs = 250

qMaxDepth = 6

guiStockfishDepth = 10
guiStockfishTimeMs = 0
