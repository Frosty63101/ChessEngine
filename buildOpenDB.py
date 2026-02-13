import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import chess.pgn
from tqdm import tqdm

# ---------- Data structures ----------


@dataclass
class GameMeta:
    resultKey: str  # "1-0", "0-1", "1/2-1/2", or "*"
    whiteElo: int
    blackElo: int


@dataclass
class MoveAgg:
    # Aggregation for a (positionKey, moveUci)
    positionKey: str
    moveUci: str
    san: str
    count: int
    whiteWins: int
    draws: int
    blackWins: int
    sumWhiteElo: int
    sumBlackElo: int


# ---------- Helpers ----------

def safeInt(value: Optional[str], defaultValue: int = 0) -> int:
    """
    Tries to parse an int from a string; returns defaultValue if missing/invalid.
    """
    if value is None:
        return defaultValue
    try:
        return int(value)
    except ValueError:
        return defaultValue


def normalizeFen4(board: chess.Board) -> str:
    """
    Returns the first 4 FEN fields:
      board, side-to-move, castling, en-passant

    This intentionally ignores halfmove/fullmove counters so transpositions merge.
    """
    fenParts = board.fen().split(" ")
    # fenParts = [board, turn, castling, ep, halfmove, fullmove]
    return " ".join(fenParts[:4])


def resultToWdl(resultKey: str) -> Tuple[int, int, int]:
    """
    Converts PGN Result header to (whiteWins, draws, blackWins) increments.
    """
    if resultKey == "1-0":
        return (1, 0, 0)
    if resultKey == "0-1":
        return (0, 0, 1)
    if resultKey == "1/2-1/2":
        return (0, 1, 0)
    return (0, 0, 0)  # unknown or "*"


def iterPgnFiles(inputDir: str) -> Iterable[str]:
    """
    Yields all .pgn file paths under inputDir (non-recursive).
    If you want recursive, switch os.listdir -> os.walk.
    """
    for fileName in os.listdir(inputDir):
        if fileName.lower().endswith(".pgn"):
            yield os.path.join(inputDir, fileName)


# ---------- SQLite setup ----------

def createSchema(conn: sqlite3.Connection) -> None:
    """
    Creates tables + indexes. Uses WAL and other pragmas for speed.
    """
    cur = conn.cursor()

    # Speed pragmas (good for bulk build; less safe for power loss)
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    # Negative cache_size = KB units. -200000 = ~200MB cache.
    cur.execute("PRAGMA cache_size=-200000;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        positionKey TEXT PRIMARY KEY,
        fen4 TEXT NOT NULL,
        ply INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS moves (
        positionKey TEXT NOT NULL,
        moveUci TEXT NOT NULL,
        san TEXT NOT NULL,
        count INTEGER NOT NULL,
        whiteWins INTEGER NOT NULL,
        draws INTEGER NOT NULL,
        blackWins INTEGER NOT NULL,
        sumWhiteElo INTEGER NOT NULL,
        sumBlackElo INTEGER NOT NULL,
        PRIMARY KEY (positionKey, moveUci),
        FOREIGN KEY (positionKey) REFERENCES positions(positionKey)
    );
    """)

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_moves_position ON moves(positionKey);")
    conn.commit()


def upsertPositions(conn: sqlite3.Connection, positionRows: List[Tuple[str, str, int]]) -> None:
    """
    Inserts positions; ignores if already present.
    positionRows: [(positionKey, fen4, ply), ...]
    """
    if not positionRows:
        return

    conn.executemany("""
    INSERT INTO positions(positionKey, fen4, ply)
    VALUES (?, ?, ?)
    ON CONFLICT(positionKey) DO NOTHING;
    """, positionRows)


def upsertMoves(conn: sqlite3.Connection, moveRows: List[Tuple[str, str, str, int, int, int, int, int, int]]) -> None:
    """
    Upserts aggregated move stats.
    moveRows tuple order matches columns.
    """
    if not moveRows:
        return

    conn.executemany("""
    INSERT INTO moves(
        positionKey, moveUci, san,
        count, whiteWins, draws, blackWins,
        sumWhiteElo, sumBlackElo
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(positionKey, moveUci) DO UPDATE SET
        san = excluded.san,
        count = moves.count + excluded.count,
        whiteWins = moves.whiteWins + excluded.whiteWins,
        draws = moves.draws + excluded.draws,
        blackWins = moves.blackWins + excluded.blackWins,
        sumWhiteElo = moves.sumWhiteElo + excluded.sumWhiteElo,
        sumBlackElo = moves.sumBlackElo + excluded.sumBlackElo;
    """, moveRows)


# ---------- PGN processing ----------

def readGameMeta(game: chess.pgn.Game) -> GameMeta:
    """
    Extracts key metadata from headers.
    """
    headers = game.headers
    resultKey = headers.get("Result", "*")
    whiteElo = safeInt(headers.get("WhiteElo"), 0)
    blackElo = safeInt(headers.get("BlackElo"), 0)
    return GameMeta(resultKey=resultKey, whiteElo=whiteElo, blackElo=blackElo)


def processSingleGame(
    game: chess.pgn.Game,
    maxPlies: int,
    minElo: int,
) -> Tuple[List[Tuple[str, str, int]], Dict[Tuple[str, str], MoveAgg]]:
    """
    Processes one PGN game and returns:
      - positionsRows to insert (positionKey, fen4, ply)
      - moveAgg dict keyed by (positionKey, moveUci)

    This only records the first maxPlies plies from the mainline.
    """
    meta = readGameMeta(game)

    # Optional quality filter: ignore games with unknown/low Elo
    if (meta.whiteElo < minElo) or (meta.blackElo < minElo):
        return ([], {})

    whiteWinsInc, drawsInc, blackWinsInc = resultToWdl(meta.resultKey)

    board = game.board()
    positionsRows: List[Tuple[str, str, int]] = []
    moveAggs: Dict[Tuple[str, str], MoveAgg] = {}

    ply = 0
    for move in game.mainline_moves():
        if ply >= maxPlies:
            break

        fen4 = normalizeFen4(board)
        positionKey = fen4  # fen4 is already a solid key for transpositions

        # record the position (ply is useful for debugging / filtering)
        positionsRows.append((positionKey, fen4, ply))

        # SAN must be computed before pushing move
        try:
            san = board.san(move)
        except ValueError:
            # Corrupt/illegal move relative to board; skip this game safely
            break

        moveUci = move.uci()
        aggKey = (positionKey, moveUci)

        if aggKey not in moveAggs:
            moveAggs[aggKey] = MoveAgg(
                positionKey=positionKey,
                moveUci=moveUci,
                san=san,
                count=0,
                whiteWins=0,
                draws=0,
                blackWins=0,
                sumWhiteElo=0,
                sumBlackElo=0,
            )

        agg = moveAggs[aggKey]
        agg.count += 1
        agg.whiteWins += whiteWinsInc
        agg.draws += drawsInc
        agg.blackWins += blackWinsInc
        agg.sumWhiteElo += meta.whiteElo
        agg.sumBlackElo += meta.blackElo

        board.push(move)
        ply += 1

    return (positionsRows, moveAggs)


def processPgnFile(
    filePath: str,
    maxPlies: int,
    minElo: int,
) -> Tuple[List[Tuple[str, str, int]], Dict[Tuple[str, str], MoveAgg], int]:
    """
    Streams games from a PGN file and aggregates positions/moves for that file.
    Returns (positionsRows, moveAggs, gamesProcessed)
    """
    positionsRowsAll: List[Tuple[str, str, int]] = []
    moveAggsAll: Dict[Tuple[str, str], MoveAgg] = {}
    gamesProcessed = 0

    # Use latin-1 as a “don’t crash” default; PGNs sometimes contain weird chars.
    with open(filePath, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            gamesProcessed += 1
            positionsRows, moveAggs = processSingleGame(game, maxPlies, minElo)

            # Merge positions (just append; DB will de-dupe via ON CONFLICT DO NOTHING)
            positionsRowsAll.extend(positionsRows)

            # Merge move aggs
            for key, agg in moveAggs.items():
                if key not in moveAggsAll:
                    moveAggsAll[key] = agg
                else:
                    existing = moveAggsAll[key]
                    existing.count += agg.count
                    existing.whiteWins += agg.whiteWins
                    existing.draws += agg.draws
                    existing.blackWins += agg.blackWins
                    existing.sumWhiteElo += agg.sumWhiteElo
                    existing.sumBlackElo += agg.sumBlackElo
                    # Keep the latest SAN we saw (doesn’t really matter)
                    existing.san = agg.san

    return (positionsRowsAll, moveAggsAll, gamesProcessed)


# ---------- Main build loop ----------

def buildOpeningDatabase(
    inputDir: str,
    outputDbPath: str,
    maxPlies: int,
    minElo: int,
    commitEveryFiles: int,
) -> None:
    """
    Walks all PGN files, streams games, and writes aggregated stats to SQLite in batches.
    """
    pgnFiles = list(iterPgnFiles(inputDir))
    if not pgnFiles:
        print(f"No .pgn files found in: {inputDir}")
        return

    os.makedirs(os.path.dirname(outputDbPath) or ".", exist_ok=True)

    conn = sqlite3.connect(outputDbPath)
    try:
        createSchema(conn)

        totalGames = 0
        startTime = time.time()

        pendingPositions: List[Tuple[str, str, int]] = []
        pendingMoves: Dict[Tuple[str, str], MoveAgg] = {}

        def flushBatch() -> None:
            """
            Writes pending positions + moves to the DB in a single transaction.
            """
            if not pendingPositions and not pendingMoves:
                return

            with conn:  # transaction
                upsertPositions(conn, pendingPositions)

                moveRows = [
                    (
                        agg.positionKey,
                        agg.moveUci,
                        agg.san,
                        agg.count,
                        agg.whiteWins,
                        agg.draws,
                        agg.blackWins,
                        agg.sumWhiteElo,
                        agg.sumBlackElo,
                    )
                    for agg in pendingMoves.values()
                ]
                upsertMoves(conn, moveRows)

            pendingPositions.clear()
            pendingMoves.clear()

        for fileIndex, filePath in enumerate(tqdm(pgnFiles, desc="Processing PGN files"), start=1):
            positionsRows, moveAggs, gamesProcessed = processPgnFile(
                filePath=filePath,
                maxPlies=maxPlies,
                minElo=minElo,
            )
            totalGames += gamesProcessed

            pendingPositions.extend(positionsRows)

            # Merge file moveAggs into pendingMoves so we can upsert fewer rows
            for key, agg in moveAggs.items():
                if key not in pendingMoves:
                    pendingMoves[key] = agg
                else:
                    existing = pendingMoves[key]
                    existing.count += agg.count
                    existing.whiteWins += agg.whiteWins
                    existing.draws += agg.draws
                    existing.blackWins += agg.blackWins
                    existing.sumWhiteElo += agg.sumWhiteElo
                    existing.sumBlackElo += agg.sumBlackElo
                    existing.san = agg.san

            if fileIndex % commitEveryFiles == 0:
                flushBatch()

        # final flush
        flushBatch()

        elapsed = time.time() - startTime
        print(f"\nDone.")
        print(f"DB: {outputDbPath}")
        print(f"Files processed: {len(pgnFiles)}")
        print(f"Games processed: {totalGames}")
        print(f"Elapsed: {elapsed:.1f}s")

    finally:
        conn.close()


def parseArgs(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an opening database (SQLite) from a folder of PGN files."
    )
    parser.add_argument(
        "--inputDir",
        default=r"D:\ChessEngine\pgns_small",
        help="Folder containing .pgn files (default: D:\\ChessEngine\\pgns_small)",
    )
    parser.add_argument(
        "--outputDb",
        default=r"D:\ChessEngine\opening_book.sqlite",
        help="SQLite output path",
    )
    parser.add_argument(
        "--maxPlies",
        type=int,
        default=20,
        help="How many plies (half-moves) to store per game (default: 20 = 10 moves)",
    )
    parser.add_argument(
        "--minElo",
        type=int,
        default=0,
        help="Ignore games where either player Elo is below this (default: 0 = no filter)",
    )
    parser.add_argument(
        "--commitEveryFiles",
        type=int,
        default=10,
        help="Commit to SQLite every N files (default: 10). Increase for speed, decrease for safety.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parseArgs(sys.argv[1:])
    buildOpeningDatabase(
        inputDir=args.inputDir,
        outputDbPath=args.outputDb,
        maxPlies=args.maxPlies,
        minElo=args.minElo,
        commitEveryFiles=args.commitEveryFiles,
    )


if __name__ == "__main__":
    main()
