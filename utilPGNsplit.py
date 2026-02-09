import argparse
import os

import chess.pgn


def splitSinglePgn(inputPath: str, outputDir: str, gamesPerChunk: int) -> int:
    """
    Splits one PGN into multiple smaller PGN files.

    - inputPath: path to a large .pgn
    - outputDir: where chunks go
    - gamesPerChunk: how many games per output file

    Returns: number of chunk files written.
    """
    os.makedirs(outputDir, exist_ok=True)

    baseName = os.path.splitext(os.path.basename(inputPath))[0]

    chunkIndex = 1
    gamesInChunk = 0
    totalGames = 0
    outFileHandle = None

    def openNewChunk():
        nonlocal outFileHandle, gamesInChunk, chunkIndex
        if outFileHandle is not None:
            outFileHandle.close()
        gamesInChunk = 0
        chunkPath = os.path.join(outputDir, f"{baseName}_part{chunkIndex:04d}.pgn")
        outFileHandle = open(chunkPath, "w", encoding="utf-8")
        chunkIndex += 1

    openNewChunk()

    with open(inputPath, "r", encoding="utf-8", errors="ignore") as inFileHandle:
        while True:
            game = chess.pgn.read_game(inFileHandle)
            if game is None:
                break

            # Start a new output file once we hit the limit
            if gamesInChunk >= gamesPerChunk:
                openNewChunk()

            exporter = chess.pgn.FileExporter(outFileHandle)
            game.accept(exporter)

            # Ensure a blank line between games (common PGN style)
            outFileHandle.write("\n\n")

            gamesInChunk += 1
            totalGames += 1

            if totalGames % 50 == 0:
                print(f"[split] {os.path.basename(inputPath)}: wrote {totalGames} games...", flush=True)

    if outFileHandle is not None:
        outFileHandle.close()

    # chunkIndex was incremented after last open, so chunks written = chunkIndex - 1
    chunksWritten = chunkIndex - 1
    print(f"[split] done: {os.path.basename(inputPath)} -> {chunksWritten} chunk file(s), total games={totalGames}", flush=True)
    return chunksWritten


def splitAllPgns(inputDir: str, outputDir: str, gamesPerChunk: int):
    """
    Splits every .pgn inside inputDir into chunked .pgn files in outputDir.
    """
    os.makedirs(outputDir, exist_ok=True)

    pgnPaths = [
        os.path.join(inputDir, name)
        for name in os.listdir(inputDir)
        if name.lower().endswith(".pgn")
    ]

    if not pgnPaths:
        print(f"[split] no .pgn files found in {inputDir}")
        return

    print(f"[split] found {len(pgnPaths)} PGN file(s) in {inputDir}")
    for pgnPath in sorted(pgnPaths):
        splitSinglePgn(pgnPath, outputDir, gamesPerChunk)


def main():
    parser = argparse.ArgumentParser(description="Split large PGN files into smaller chunk PGNs (by game count).")
    parser.add_argument("--inputDir", required=True, help="Directory containing large .pgn files")
    parser.add_argument("--outputDir", required=True, help="Directory to write chunked .pgn files")
    parser.add_argument("--gamesPerChunk", type=int, default=200, help="How many games per output file (default 200)")
    args = parser.parse_args()

    splitAllPgns(args.inputDir, args.outputDir, args.gamesPerChunk)


if __name__ == "__main__":
    main()
