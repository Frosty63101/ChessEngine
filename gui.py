"""Tkinter GUI for playing and training the chess evaluation model."""

import json
import math
import os
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from tkinter import messagebox

import chess
import chess.engine
from PIL import Image, ImageTk

from config import (defaultSearchDepth, defaultThinkTimeMs, guiStockfishDepth,
                    guiStockfishTimeMs, imageDir, modelEloPath, weightsPath)
from engine import AlphaBetaEngine
from model import EvalModel, loadModel, saveModel
from trainer import trainEvalModel


class EvalBar(tk.Canvas):
    """
    Chess.com-like evaluation bar:
      - White advantage fills from the top.
      - Black advantage fills from the bottom.
      - Uses a non-linear logistic mapping (so it saturates like real eval bars).
      - Can display mate as M# when provided.
    """

    def __init__(self, parent, width=30, height=220, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.width = int(width)
        self.height = int(height)
        self.maxAbs = 10.0
        self.logisticK = 0.55
        self.evalValue = 0.0
        self.mateIn = None
        self.draw()

    def setEval(self, evalPawns: float, mateIn: int | None = None):
        self.mateIn = mateIn
        if evalPawns > self.maxAbs:
            evalPawns = self.maxAbs
        if evalPawns < -self.maxAbs:
            evalPawns = -self.maxAbs
        self.evalValue = float(evalPawns)
        self.draw()

    def _whiteFractionFromEval(self) -> float:
        if isinstance(self.mateIn, int):
            # mateIn > 0 => White mates, mateIn < 0 => Black mates, mateIn == 0 => game already decided
            if self.mateIn > 0:
                return 1.0
            if self.mateIn < 0:
                return 0.0
            # mateIn == 0: leave bar fully to the side implied by evalValue sign
            return 1.0 if self.evalValue > 0 else 0.0

        x = self.evalValue
        try:
            frac = 1.0 / (1.0 + math.exp(-self.logisticK * x))
        except OverflowError:
            frac = 1.0 if x > 0 else 0.0
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        return frac

    def _formatEvalText(self) -> str:
        if isinstance(self.mateIn, int):
            return f"M{abs(self.mateIn)}"
        return f"{self.evalValue:+.2f}"

    def draw(self):
        self.delete("all")
        self.create_rectangle(0, 0, self.width, self.height,
                              fill="gray20", outline="gray50")
        whiteFrac = self._whiteFractionFromEval()
        whiteHeight = int(self.height * whiteFrac)
        boundaryY = self.height - whiteHeight
        self.create_rectangle(0, 0, self.width, boundaryY,
                              fill="black", outline="")
        self.create_rectangle(0, boundaryY, self.width,
                              self.height, fill="white", outline="")
        midY = self.height // 2
        self.create_line(0, midY, self.width, midY, fill="gray55")
        label = self._formatEvalText()
        centerY = self.height // 2
        centerOnWhite = centerY >= boundaryY
        textColor = "black" if centerOnWhite else "white"
        self.create_text(self.width // 2, self.height //
                         2, fill=textColor, text=label)


class ChessGuiApp:
    """Main GUI controller that wires UI, engine, model, and Stockfish."""

    def __init__(self, root: tk.Tk, stockfishPath: str, initialModel: EvalModel | None):
        self.root = root
        self.stockfishPath = stockfishPath
        self.evalThreadRunning = False
        self.model = initialModel if initialModel is not None else self._defaultModel()
        self.engine = AlphaBetaEngine(
            self.model, openingDbPath=r"D:\ChessEngine\opening_book.sqlite")
        self.board = chess.Board()
        self.moveThreadRunning = False
        self.moveThreadLock = threading.Lock()
        self.selectedSquare = None
        self.autoPlayVar = tk.BooleanVar(value=False)
        self.searchDepthVar = tk.IntVar(value=defaultSearchDepth)
        self.thinkTimeVar = tk.IntVar(value=defaultThinkTimeMs)
        self.mode = "humanVsEngine"
        self.modelPlaysWhiteVar = tk.BooleanVar(value=True)
        self.fishPlayer = chess.engine.SimpleEngine.popen_uci(
            self.stockfishPath)
        self.fish = chess.engine.SimpleEngine.popen_uci(self.stockfishPath)
        self.sfSkillVar = tk.IntVar(value=10)
        self.sfUseEloVar = tk.BooleanVar(value=False)
        self.sfEloVar = tk.IntVar(value=1350)

        # NEW: training stop controls
        self.trainingActive = False
        self.trainingStopRequested = False

        self.matchRunnerActive = False
        self.matchCancelRequested = False
        self.eloSimActive = False
        self.eloSimCancelRequested = False
        self.modelElo = None

        self.applyStockfishStrengthToEngine(self.fishPlayer)
        self._loadPieceImages()
        self._buildUi()
        self._loadModelElo()
        self.drawBoard()
        self.updateEvalBarsAsync()

    def _defaultModel(self):
        w = [0.0] * 16
        w[1] = 1.0
        w[2] = 3.2
        w[3] = 3.3
        w[4] = 5.1
        w[5] = 9.2
        return EvalModel(weights=self._toNp(w))

    def _toNp(self, w):
        import numpy as np
        return np.array(w, dtype=np.float64)

    def _loadPieceImages(self):
        self.pieceImages = {}
        for piece in ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]:
            if piece.isupper():
                filename = os.path.join(imageDir, f"{piece}.png")
            else:
                filename = os.path.join(imageDir, f"_{piece}.png")
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Missing piece image: {filename}")
            img = Image.open(filename).resize(
                (50, 50), Image.Resampling.LANCZOS)
            self.pieceImages[piece] = img

    def _buildUi(self):
        self.root.title("Learned Eval + Alpha-Beta Engine (Upgraded)")
        self.root.geometry("1200x700")

        leftFrame = tk.Frame(self.root)
        leftFrame.pack(side=tk.LEFT, padx=10, pady=10)
        rightFrame = tk.Frame(self.root)
        rightFrame.pack(side=tk.RIGHT, fill=tk.BOTH,
                        expand=True, padx=10, pady=10)

        self.boardCanvas = tk.Canvas(leftFrame, width=400, height=400)
        self.boardCanvas.pack()
        self.boardCanvas.bind("<Button-1>", self.onBoardClick)

        evalFrame = tk.LabelFrame(leftFrame, text="Eval Bars")
        evalFrame.pack(fill=tk.X, pady=8)

        self.modelEvalBar = EvalBar(evalFrame)
        self.modelEvalBar.pack(side=tk.LEFT, padx=10)
        tk.Label(evalFrame, text="Model", fg="white",
                 bg=self.root.cget("bg")).pack(side=tk.LEFT, padx=5)

        self.sfEvalBar = EvalBar(evalFrame)
        self.sfEvalBar.pack(side=tk.LEFT, padx=10)
        tk.Label(evalFrame, text="Stockfish", fg="white",
                 bg=self.root.cget("bg")).pack(side=tk.LEFT, padx=5)

        ctrlFrame = tk.LabelFrame(rightFrame, text="Controls")
        ctrlFrame.pack(fill=tk.X, pady=5)

        tk.Label(ctrlFrame, text="Search depth:").grid(
            row=0, column=0, sticky="w")
        tk.Spinbox(ctrlFrame, from_=1, to=16, textvariable=self.searchDepthVar,
                   width=5).grid(row=0, column=1, sticky="w")

        tk.Label(ctrlFrame, text="Think time (ms):").grid(
            row=1, column=0, sticky="w")
        tk.Spinbox(ctrlFrame, from_=50, to=5000, increment=50,
                   textvariable=self.thinkTimeVar, width=7).grid(row=1, column=1, sticky="w")

        tk.Button(ctrlFrame, text="New Game (Human vs Engine)",
                  command=self.newGameHumanVsEngine).grid(row=0, column=2, padx=5)
        tk.Button(ctrlFrame, text="New Game (Engine vs Engine)",
                  command=self.newGameEngineVsEngine).grid(row=1, column=2, padx=5)
        tk.Button(ctrlFrame, text="New Game (Model vs Stockfish)",
                  command=self.newGameModelVsStockfish).grid(row=2, column=2, padx=5)

        tk.Checkbutton(ctrlFrame, text="Model plays White (else Black)",
                       variable=self.modelPlaysWhiteVar).grid(row=2, column=3, padx=5, sticky="w")

        # --- Stockfish strength frame (unchanged) ---
        sfFrame = tk.LabelFrame(
            rightFrame, text="Stockfish Strength (for playing)")
        sfFrame.pack(fill=tk.X, pady=5)

        tk.Label(sfFrame, text="Skill (0-20):").grid(row=0,
                                                     column=0, sticky="w")
        self.sfSkillScale = tk.Scale(sfFrame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.sfSkillVar,
                                     command=lambda _: self.onStockfishStrengthChanged())
        self.sfSkillScale.grid(row=0, column=1, sticky="we", padx=5)
        sfFrame.grid_columnconfigure(1, weight=1)

        self.sfUseEloCheck = tk.Checkbutton(sfFrame, text="Limit by Elo", variable=self.sfUseEloVar,
                                            command=self.onStockfishStrengthChanged)
        self.sfUseEloCheck.grid(row=1, column=0, sticky="w")

        tk.Label(sfFrame, text="Elo:").grid(row=1, column=1, sticky="w")
        self.sfEloScale = tk.Scale(sfFrame, from_=800, to=2850, resolution=50, orient=tk.HORIZONTAL,
                                   variable=self.sfEloVar, command=lambda _: self.onStockfishStrengthChanged())
        self.sfEloScale.grid(
            row=2, column=0, columnspan=2, sticky="we", padx=5)

        tk.Button(sfFrame, text="Apply Strength", command=self.onStockfishStrengthChanged).grid(
            row=3, column=0, pady=4, sticky="w")

        # --- Match Runner + Elo Simulation frames (unchanged in your pasted version) ---
        matchFrame = tk.LabelFrame(
            rightFrame, text="Match Runner (Model vs Stockfish)")
        matchFrame.pack(fill=tk.X, pady=5)
        tk.Label(matchFrame, text="Games:").grid(row=0, column=0, sticky="w")
        self.matchGamesEntry = tk.Entry(matchFrame, width=8)
        self.matchGamesEntry.insert(0, "20")
        self.matchGamesEntry.grid(row=0, column=1, sticky="w")
        tk.Label(matchFrame, text="Max plies/game:").grid(row=0,
                                                          column=2, sticky="w", padx=(10, 0))
        self.matchMaxPliesEntry = tk.Entry(matchFrame, width=8)
        self.matchMaxPliesEntry.insert(0, "200")
        self.matchMaxPliesEntry.grid(row=0, column=3, sticky="w")
        self.matchAlternateColorsVar = tk.BooleanVar(value=True)
        tk.Checkbutton(matchFrame, text="Alternate colors", variable=self.matchAlternateColorsVar).grid(
            row=1, column=0, columnspan=2, sticky="w")
        self.matchVerboseVar = tk.BooleanVar(value=False)
        tk.Checkbutton(matchFrame, text="Verbose (print each result)", variable=self.matchVerboseVar).grid(
            row=1, column=2, columnspan=2, sticky="w")
        tk.Button(matchFrame, text="Run Matches (background)",
                  command=self.runMatchesInBackground).grid(row=2, column=0, pady=4, sticky="w")
        tk.Button(matchFrame, text="Cancel Matches", command=self.cancelMatches).grid(
            row=2, column=1, pady=4, sticky="w")

        eloFrame = tk.LabelFrame(rightFrame, text="Model Elo Simulation")
        eloFrame.pack(fill=tk.X, pady=5)
        self.modelEloLabel = tk.Label(
            eloFrame, text="Current simulated Elo: N/A", anchor="w")
        self.modelEloLabel.grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Label(eloFrame, text="Opponent Elos:").grid(
            row=1, column=0, sticky="w")
        self.eloListEntry = tk.Entry(eloFrame, width=26)
        self.eloListEntry.insert(0, "800,1000,1200,1400,1600,1800")
        self.eloListEntry.grid(row=1, column=1, columnspan=3, sticky="w")
        tk.Label(eloFrame, text="Games per Elo:").grid(
            row=2, column=0, sticky="w")
        self.eloGamesEntry = tk.Entry(eloFrame, width=8)
        self.eloGamesEntry.insert(0, "6")
        self.eloGamesEntry.grid(row=2, column=1, sticky="w")
        tk.Label(eloFrame, text="Max plies/game:").grid(row=2,
                                                        column=2, sticky="w", padx=(10, 0))
        self.eloMaxPliesEntry = tk.Entry(eloFrame, width=8)
        self.eloMaxPliesEntry.insert(0, "200")
        self.eloMaxPliesEntry.grid(row=2, column=3, sticky="w")
        tk.Button(eloFrame, text="Run Elo Simulation (background)",
                  command=self.runEloSimulationInBackground).grid(row=3, column=0, pady=4, sticky="w")
        tk.Button(eloFrame, text="Cancel Elo Simulation", command=self.cancelEloSimulation).grid(
            row=3, column=1, pady=4, sticky="w")

        tk.Button(ctrlFrame, text="Engine Move Now",
                  command=self.engineMoveNow).grid(row=0, column=3, padx=5)
        tk.Checkbutton(ctrlFrame, text="Auto-play self game", variable=self.autoPlayVar,
                       command=self.onToggleAutoplay).grid(row=1, column=3, padx=5)

        # --- Training frame (UPDATED) ---
        trainFrame = tk.LabelFrame(
            rightFrame, text="Training (self-play + PGN + random → fit to Stockfish)")
        trainFrame.pack(fill=tk.X, pady=5)

        self.sampleEntry = tk.Entry(trainFrame, width=10)
        self.sampleEntry.insert(0, "5000")
        self.sampleEntry.grid(row=0, column=1, sticky="w")
        tk.Label(trainFrame, text="Samples:").grid(row=0, column=0, sticky="w")

        self.depthEntry = tk.Entry(trainFrame, width=10)
        self.depthEntry.insert(0, "10")
        self.depthEntry.grid(row=1, column=1, sticky="w")
        tk.Label(trainFrame, text="Stockfish depth:").grid(
            row=1, column=0, sticky="w")

        tk.Button(trainFrame, text="Train (background)",
                  command=self.trainInBackground).grid(row=0, column=2, padx=5)

        # NEW: stop sample collection early
        self.stopCollectButton = tk.Button(
            trainFrame,
            text="Stop sample collection",
            command=self.stopSampleCollection,
            state=tk.DISABLED
        )
        self.stopCollectButton.grid(row=0, column=3, padx=5)

        tk.Button(trainFrame, text="Save Weights",
                  command=self.saveWeights).grid(row=1, column=2, padx=5)
        tk.Button(trainFrame, text="Load Weights",
                  command=self.loadWeights).grid(row=2, column=2, padx=5)

        self.statusLabel = tk.Label(
            rightFrame, text="Status: ready", anchor="w")
        self.statusLabel.pack(fill=tk.X)

        self.moveList = tk.Text(rightFrame, height=18)
        self.moveList.pack(fill=tk.BOTH, expand=True)
        self._resetMoveList()

        self.root.protocol("WM_DELETE_WINDOW", self.onClose)

    def stopSampleCollection(self):
        if not self.trainingActive:
            self.setStatus("training: not active")
            return
        self.trainingStopRequested = True
        self.setStatus(
            "training: stop requested (will fit with collected samples)")

    def _squashToModelScale(self, rawEvalPawns: float) -> float:
        """
        Apply the same tanh squash used during training so Stockfish and the model
        are displayed on the same scale.
        """
        scale = float(getattr(self.model, "labelScale", 0.0) or 0.0)
        if scale <= 0.0:
            return float(rawEvalPawns)
        return float(scale * math.tanh(float(rawEvalPawns) / scale))

    def onClose(self):
        try:
            self.fish.quit()
        except Exception:
            pass
        try:
            self.fishPlayer.quit()
        except Exception:
            pass
        self.root.destroy()

    def _resetMoveList(self):
        self.moveList.delete("1.0", tk.END)
        self.moveList.insert(tk.END, "Moves:\n")

    def setStatus(self, text: str):
        self.statusLabel.config(text=f"Status: {text}")
        self.root.update_idletasks()

    def drawBoard(self):
        """Render the current board position and selection highlight."""
        boardImage = Image.new("RGB", (400, 400), "white")
        squareSize = 50
        for rank in range(8):
            for file in range(8):
                color = "lightgray" if (rank + file) % 2 == 0 else "darkgreen"
                x0 = file * squareSize
                y0 = rank * squareSize
                x1 = x0 + squareSize
                y1 = y0 + squareSize
                boardImage.paste(color, (x0, y0, x1, y1))
                square = chess.square(file, 7 - rank)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = piece.symbol()
                    pieceImg = self.pieceImages.get(symbol)
                    if pieceImg:
                        boardImage.paste(pieceImg, (x0, y0), pieceImg)
        boardTk = ImageTk.PhotoImage(boardImage)
        self.boardCanvas.create_image(0, 0, anchor="nw", image=boardTk)
        self.boardCanvas.image = boardTk
        self.boardCanvas.delete("highlight")
        if self.selectedSquare is not None:
            sqSize = 50
            file = chess.square_file(self.selectedSquare)
            rank = 7 - chess.square_rank(self.selectedSquare)
            x0 = file * sqSize
            y0 = rank * sqSize
            x1 = x0 + sqSize
            y1 = y0 + sqSize
            self.boardCanvas.create_rectangle(
                x0, y0, x1, y1, outline="red", width=2, tags="highlight")

    def updateEvalBarsAsync(self):
        """Recompute model/Stockfish evaluations on a background thread."""
        if self.evalThreadRunning:
            return
        self.evalThreadRunning = True
        fen = self.board.fen()

        def worker():
            modelEvalWhitePov = 0.0
            sfEval = 0.0
            mateIn = None

            try:
                boardCopy = chess.Board(fen)

                modelEvalWhitePov = float(self.engine.evaluate(boardCopy))

                print(
                    "turn=", "W" if boardCopy.turn == chess.WHITE else "B",
                    "modelEvalWhitePov=", round(modelEvalWhitePov, 2),
                    flush=True
                )

                try:
                    if guiStockfishTimeMs and guiStockfishTimeMs > 0:
                        info = self.fish.analyse(
                            boardCopy, chess.engine.Limit(
                                time=guiStockfishTimeMs / 1000.0)
                        )
                    else:
                        info = self.fish.analyse(
                            boardCopy, chess.engine.Limit(
                                depth=guiStockfishDepth)
                        )

                    score = info["score"].pov(chess.WHITE)

                    if score.is_mate():
                        mateIn = score.mate()

                        # mateIn == 0 means "checkmate already in this position"
                        if mateIn == 0:
                            if boardCopy.is_checkmate():
                                # If it's checkmate and it's White to move, White is mated => bad for White
                                sfEvalRaw = -100.0 if boardCopy.turn == chess.WHITE else 100.0
                            else:
                                sfEvalRaw = 0.0
                        else:
                            sfEvalRaw = 100.0 if (
                                mateIn is not None and mateIn > 0) else -100.0

                        # squash to model scale (and clamp for the bar)
                        sfEval = self._squashToModelScale(sfEvalRaw)

                    else:
                        cp = score.score(mate_score=100000)
                        sfEvalRaw = 0.0 if cp is None else float(cp) / 100.0
                        sfEval = self._squashToModelScale(sfEvalRaw)

                except Exception as exc:
                    print(f"[sf] analyse failed: {exc}", flush=True)
                    sfEval = 0.0
                    mateIn = None

            finally:
                self.evalThreadRunning = False

            def apply():
                # Display both as White POV so they are directly comparable
                self.modelEvalBar.setEval(modelEvalWhitePov)
                self.sfEvalBar.setEval(sfEval, mateIn=mateIn)
                print(
                    f"Model eval (White POV): {modelEvalWhitePov:.2f} | "
                    f"Stockfish eval (White POV): {sfEval:.2f} (mate in {mateIn})",
                    flush=True
                )

            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def onBoardClick(self, event):
        """Handle human move selection for the human-vs-engine mode."""
        if self.mode != "humanVsEngine":
            return
        if self.board.turn != chess.WHITE:
            return
        file = event.x // 50
        rank = event.y // 50
        actualRank = 7 - rank
        square = chess.square(file, actualRank)
        if self.selectedSquare is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selectedSquare = square
                self.drawBoard()
            return
        fromSq = self.selectedSquare
        toSq = square
        self.selectedSquare = None
        move = chess.Move(fromSq, toSq)
        movingPiece = self.board.piece_at(fromSq)
        if movingPiece and movingPiece.piece_type == chess.PAWN:
            promoRank = 7 if movingPiece.color == chess.WHITE else 0
            if chess.square_rank(toSq) == promoRank:
                move = chess.Move(fromSq, toSq, promotion=chess.QUEEN)
        if move not in self.board.legal_moves:
            self.drawBoard()
            return
        san = self.board.san(move)
        self.board.push(move)
        self._appendMoveSan(san)
        self.drawBoard()
        self.updateEvalBarsAsync()
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", f"Result: {self.board.result()}")
            return
        self.engineMoveNow()

    def _appendMoveSan(self, san: str):
        """Append a SAN move to the move list widget."""
        moveCount = self.board.fullmove_number
        if self.board.turn == chess.BLACK:
            self.moveList.insert(tk.END, f"{moveCount}. {san} ")
        else:
            self.moveList.insert(tk.END, f"{san}\n")
        self.moveList.see(tk.END)

    def engineMoveNow(self):
        """Kick off a move for the current side to play (thread-safe, ignores stale results)."""
        if self.board.is_game_over():
            return

        with self.moveThreadLock:
            if self.moveThreadRunning:
                # Prevent overlapping move workers (double-click / autoplay spam)
                return
            self.moveThreadRunning = True

        print("engineMoveNow: mode=", self.mode, "turn=",
              "White" if self.board.turn == chess.WHITE else "Black")

        def releaseMoveLock():
            with self.moveThreadLock:
                self.moveThreadRunning = False

        # Special handling for model-vs-Stockfish mode
        if self.mode == "modelVsStockfish":
            try:
                if self._isStockfishTurn():
                    self._stockfishMoveNow(
                        continueAutoplay=False, onDone=releaseMoveLock)
                else:
                    self._modelMoveNow(continueAutoplay=False,
                                       onDone=releaseMoveLock)
            except Exception:
                releaseMoveLock()
            return

        depth = int(self.searchDepthVar.get())
        thinkMs = int(self.thinkTimeVar.get())

        fenSnapshot = self.board.fen()
        self.setStatus(f"thinking (depth<= {depth}, {thinkMs}ms)...")

        def worker():
            try:
                boardCopy = chess.Board(fenSnapshot)
                result = self.engine.searchIterativeDeepening(
                    boardCopy, maxDepth=depth, timeLimitMs=thinkMs
                )

                def apply():
                    try:
                        if self.board.fen() != fenSnapshot:
                            self.setStatus(
                                "ignored engine result (position changed while thinking)")
                            return
                        self._applyEngineResult(
                            result.bestMove, result.nodes, result.bestScore, result.depth
                        )
                    finally:
                        releaseMoveLock()

                self.root.after(0, apply)

            except Exception as exc:
                self.root.after(0, lambda: (self.setStatus(
                    f"engine move failed: {exc}"), releaseMoveLock()))

        threading.Thread(target=worker, daemon=True).start()

    def _stockfishMoveNow(self, continueAutoplay: bool = False, onDone=None):
        """
        Ask Stockfish to play a move safely:
        - snapshot the current FEN
        - compute on a local board
        - apply only if position hasn't changed
        """
        self.applyStockfishStrengthToEngine(self.fishPlayer)
        self.setStatus("Stockfish thinking...")

        fenSnapshot = self.board.fen()

        def worker():
            move = None
            errText = ""

            try:
                boardCopy = chess.Board(fenSnapshot)

                if guiStockfishTimeMs and guiStockfishTimeMs > 0:
                    limit = chess.engine.Limit(
                        time=guiStockfishTimeMs / 1000.0)
                else:
                    limit = chess.engine.Limit(depth=guiStockfishDepth)

                result = self.fishPlayer.play(boardCopy, limit)
                move = result.move

            except Exception as exc:
                errText = str(exc)

            def apply():
                try:
                    if self.board.fen() != fenSnapshot:
                        self.setStatus(
                            "ignored Stockfish result (position changed while thinking)")
                        return

                    if move is None:
                        self.setStatus(f"Stockfish move failed: {errText}")
                        return

                    if move not in self.board.legal_moves:
                        self.setStatus(
                            f"Stockfish returned illegal move: {move.uci()}")
                        return

                    san = self.board.san(move)
                    self.board.push(move)
                    self._appendMoveSan(san)
                    self.drawBoard()
                    self.updateEvalBarsAsync()

                    if self.board.is_game_over():
                        messagebox.showinfo(
                            "Game Over", f"Result: {self.board.result()}")
                        self.setStatus("game over")
                        return

                    self.setStatus(f"Stockfish played {san}")

                    if continueAutoplay:
                        self.root.after(
                            120, self._autoPlayModelVsStockfishStep)
                finally:
                    if callable(onDone):
                        onDone()

            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _modelMoveNow(self, continueAutoplay: bool = False, onDone=None):
        """
        Ask the model/engine to play a move safely:
        - snapshot current FEN
        - search on a local board
        - apply only if position hasn't changed
        """
        if self.board.is_game_over():
            if callable(onDone):
                onDone()
            return

        depth = int(self.searchDepthVar.get())
        thinkMs = int(self.thinkTimeVar.get())
        self.setStatus(f"model thinking (depth<= {depth}, {thinkMs}ms)...")

        fenSnapshot = self.board.fen()

        def worker():
            try:
                boardCopy = chess.Board(fenSnapshot)
                result = self.engine.searchIterativeDeepening(
                    boardCopy, maxDepth=depth, timeLimitMs=thinkMs
                )

                bestMove = result.bestMove
                nodes = result.nodes
                bestScore = result.bestScore
                usedDepth = result.depth

                def apply():
                    try:
                        if self.board.fen() != fenSnapshot:
                            self.setStatus(
                                "ignored model result (position changed while thinking)")
                            return

                        if bestMove is None:
                            self.setStatus("model found no move")
                            return

                        if bestMove not in self.board.legal_moves:
                            self.setStatus(
                                f"model returned illegal move: {bestMove.uci()}")
                            return

                        san = self.board.san(bestMove)
                        self.board.push(bestMove)
                        self._appendMoveSan(san)
                        self.drawBoard()
                        self.updateEvalBarsAsync()

                        if self.board.is_game_over():
                            messagebox.showinfo(
                                "Game Over", f"Result: {self.board.result()}")
                            self.setStatus("game over")
                            return

                        self.setStatus(
                            f"model played {san} | nodes={nodes} | eval≈{bestScore:.2f} | depth={usedDepth}"
                        )

                        if continueAutoplay:
                            self.root.after(
                                120, self._autoPlayModelVsStockfishStep)
                        else:
                            self._maybeTriggerAIMove()
                    finally:
                        if callable(onDone):
                            onDone()

                self.root.after(0, apply)

            except Exception as exc:
                self.root.after(0, lambda: (self.setStatus(
                    f"model move failed: {exc}"), onDone() if callable(onDone) else None))

        threading.Thread(target=worker, daemon=True).start()

    def _applyEngineResult(self, move, nodes: int, score: float, depth: int):
        """Apply an engine move result and refresh UI state."""
        if move is None:
            self.setStatus("no move found")
            return

        try:
            if move not in self.board.legal_moves:
                self.setStatus(f"engine returned illegal move: {move.uci()}")
                return

            san = self.board.san(move)
            self.board.push(move)

        except Exception as exc:
            self.setStatus(f"failed applying engine move: {exc}")
            return

        self._appendMoveSan(san)
        self.drawBoard()
        self.updateEvalBarsAsync()
        self.setStatus(
            f"played {san} | nodes={nodes} | eval≈{score:.2f} pawns | depth={depth}"
        )
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", f"Result: {self.board.result()}")

    def newGameHumanVsEngine(self):
        """Reset to a fresh game where the human plays White."""
        self.mode = "humanVsEngine"
        self.autoPlayVar.set(False)
        self.board = chess.Board()
        self.selectedSquare = None
        self._resetMoveList()
        self.setStatus("new game: human (White) vs engine (Black)")
        self.drawBoard()
        self.updateEvalBarsAsync()

    def newGameEngineVsEngine(self):
        """Reset to engine-vs-engine mode and start autoplay if enabled."""
        self.mode = "engineVsEngine"
        self.board = chess.Board()
        self.selectedSquare = None
        self._resetMoveList()
        self.setStatus("new game: engine vs engine")
        self.drawBoard()
        self.updateEvalBarsAsync()
        self.onToggleAutoplay()

    def newGameModelVsStockfish(self):
        """Reset to model-vs-Stockfish mode and trigger the first move."""
        self.mode = "modelVsStockfish"
        self.autoPlayVar.set(False)
        self.board = chess.Board()
        self.selectedSquare = None
        self._resetMoveList()
        sideText = "White" if self.modelPlaysWhiteVar.get() else "Black"
        self.setStatus(f"new game: model ({sideText}) vs Stockfish")
        self.drawBoard()
        self.updateEvalBarsAsync()
        self._maybeTriggerAIMove()

    def applyStockfishStrengthToEngine(self, engine: chess.engine.SimpleEngine):
        """Configure Stockfish skill and Elo options on the given engine."""
        options = {}
        options["Skill Level"] = int(self.sfSkillVar.get())
        useElo = bool(self.sfUseEloVar.get())
        options["UCI_LimitStrength"] = useElo
        if useElo:
            options["UCI_Elo"] = int(self.sfEloVar.get())
        for key, value in options.items():
            try:
                engine.configure({key: value})
            except Exception:
                pass

    def onStockfishStrengthChanged(self):
        """Apply Stockfish strength settings and report the result."""
        try:
            self.applyStockfishStrengthToEngine(self.fishPlayer)
            skill = self.sfSkillVar.get()
            if self.sfUseEloVar.get():
                elo = self.sfEloVar.get()
                self.setStatus(
                    f"Stockfish strength applied: Skill={skill}, Elo≈{elo}")
            else:
                self.setStatus(
                    f"Stockfish strength applied: Skill={skill} (no Elo limit)")
        except Exception as exc:
            self.setStatus(f"failed to apply strength: {exc}")

    def _loadModelElo(self):
        """Load the last simulated model Elo from disk, if available."""
        if not os.path.exists(modelEloPath):
            self._setModelEloDisplay(None, None)
            return
        try:
            with open(modelEloPath, "r", encoding="utf-8") as f:
                data = json.load(f)
            rating = data.get("rating")
            updated_at = data.get("updatedAt")
            if isinstance(rating, (int, float)):
                self.modelElo = int(round(rating))
            else:
                self.modelElo = None
            self._setModelEloDisplay(self.modelElo, updated_at)
        except Exception:
            self.modelElo = None
            self._setModelEloDisplay(None, None)

    def _saveModelElo(self, rating: int, results: list[dict]):
        """Persist the simulated model Elo and raw results to disk."""
        payload = {
            "rating": int(rating),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }
        with open(modelEloPath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _setModelEloDisplay(self, rating: int | None, updated_at: str | None):
        """Update the Elo label text."""
        if rating is None:
            text = "Current simulated Elo: N/A"
        elif updated_at:
            text = f"Current simulated Elo: {rating} (updated {updated_at})"
        else:
            text = f"Current simulated Elo: {rating}"
        if hasattr(self, "modelEloLabel"):
            self.modelEloLabel.config(text=text)

    def _parseEloList(self, text: str) -> list[int]:
        """Parse a comma-separated list of Elo values."""
        raw_parts = [p.strip() for p in text.split(",")]
        elos: list[int] = []
        for part in raw_parts:
            if not part:
                continue
            value = int(part)
            if value <= 0:
                raise ValueError("Elo values must be positive")
            elos.append(value)
        if not elos:
            raise ValueError("No Elo values provided")
        return elos

    def _eloFromScore(self, opponent_elo: int, score: float) -> float:
        """Compute model Elo from an observed score vs an opponent Elo."""
        clamped = max(0.01, min(0.99, score))
        return opponent_elo - 400.0 * math.log10(1.0 / clamped - 1.0)

    def cancelEloSimulation(self):
        """Request cancellation of the Elo simulation worker."""
        self.eloSimCancelRequested = True
        self.setStatus("elo simulation: cancel requested")

    def runEloSimulationInBackground(self):
        """Simulate model Elo by playing Stockfish at multiple Elo ratings."""
        if self.eloSimActive:
            messagebox.showinfo(
                "Elo Simulation", "An Elo simulation is already active.")
            return
        try:
            elos = self._parseEloList(self.eloListEntry.get())
            games_per_elo = int(self.eloGamesEntry.get())
            max_plies = int(self.eloMaxPliesEntry.get())
            if games_per_elo <= 0 or max_plies <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Error", "Please enter valid Elo list, games per Elo, and max plies.")
            return

        self.eloSimActive = True
        self.eloSimCancelRequested = False
        self.setStatus("elo simulation: starting...")
        model_depth = int(self.searchDepthVar.get())
        model_think_ms = int(self.thinkTimeVar.get())

        if guiStockfishTimeMs and guiStockfishTimeMs > 0:
            sf_limit = chess.engine.Limit(time=guiStockfishTimeMs / 1000.0)
        else:
            sf_limit = chess.engine.Limit(depth=guiStockfishDepth)

        def worker():
            sim_fish = chess.engine.SimpleEngine.popen_uci(self.stockfishPath)
            results: list[dict] = []
            try:
                for elo in elos:
                    if self.eloSimCancelRequested:
                        break
                    try:
                        sim_fish.configure({"UCI_LimitStrength": True})
                    except Exception:
                        pass
                    try:
                        sim_fish.configure({"UCI_Elo": int(elo)})
                    except Exception:
                        pass

                    wins = 0
                    losses = 0
                    draws = 0

                    def ui_append(text: str):
                        self.root.after(0, lambda: (self.moveList.insert(
                            tk.END, text + "\n"), self.moveList.see(tk.END)))

                    ui_append(
                        f"[elo] Opponent Elo {elo} | games={games_per_elo}")

                    for game_index in range(1, games_per_elo + 1):
                        if self.eloSimCancelRequested:
                            break
                        board = chess.Board()
                        model_is_white = (game_index % 2 == 1)
                        plies_played = 0
                        while not board.is_game_over() and plies_played < max_plies:
                            if self.eloSimCancelRequested:
                                break
                            model_to_move = (board.turn == chess.WHITE) if model_is_white else (
                                board.turn == chess.BLACK)
                            if model_to_move:
                                print(
                                    f"[elo] Opponent Elo {elo} | game {game_index}/{games_per_elo} model thinking...", flush=True)
                                stime = time.perf_counter()
                                result = self.engine.searchIterativeDeepening(
                                    board, maxDepth=model_depth, timeLimitMs=model_think_ms)
                                etime = time.perf_counter()
                                print(
                                    f"[elo] Opponent Elo {elo} | game {game_index}/{games_per_elo} model move took {etime - stime:.2f}s", flush=True)
                                move = result.bestMove
                                if move is None:
                                    break
                                board.push(move)
                            else:
                                print(
                                    f"[elo] Opponent Elo {elo} | game {game_index}/{games_per_elo} thinking...", flush=True)
                                stime = time.perf_counter()
                                sf_result = sim_fish.play(board, sf_limit)
                                etime = time.perf_counter()
                                print(
                                    f"[elo] Opponent Elo {elo} | game {game_index}/{games_per_elo} Stockfish move took {etime - stime:.2f}s", flush=True)
                                if sf_result.move is None:
                                    break
                                board.push(sf_result.move)
                            plies_played += 1
                        if board.is_game_over():
                            res = board.result()
                        else:
                            res = "1/2-1/2"
                        if res == "1/2-1/2":
                            draws += 1
                        elif res == "1-0":
                            if model_is_white:
                                wins += 1
                            else:
                                losses += 1
                        elif res == "0-1":
                            if model_is_white:
                                losses += 1
                            else:
                                wins += 1
                        else:
                            draws += 1
                        print(
                            f"[elo] Opponent Elo {elo} | game {game_index}/{games_per_elo} result: {res} | wins={wins}, losses={losses}, draws={draws}", flush=True)

                    ui_append(
                        f"[elo] Opponent Elo {elo} | wins={wins}, losses={losses}, draws={draws}")

                    games_played = wins + losses + draws
                    score = 0.0 if games_played == 0 else (
                        wins + 0.5 * draws) / games_played
                    implied = self._eloFromScore(
                        elo, score) if games_played > 0 else None
                    results.append({
                        "opponentElo": int(elo),
                        "games": games_played,
                        "wins": wins,
                        "losses": losses,
                        "draws": draws,
                        "score": round(score, 4),
                        "impliedModelElo": None if implied is None else round(implied, 1),
                    })

                total_games = sum(r["games"] for r in results)
                if total_games > 0:
                    weighted_sum = 0.0
                    for r in results:
                        if r["impliedModelElo"] is None:
                            continue
                        weighted_sum += r["impliedModelElo"] * r["games"]
                    model_elo = int(round(weighted_sum / total_games)
                                    ) if weighted_sum > 0 else None
                else:
                    model_elo = None

                def apply_done():
                    if model_elo is not None:
                        self.modelElo = model_elo
                        try:
                            self._saveModelElo(model_elo, results)
                            updated_at = datetime.now(timezone.utc).isoformat()
                            self._setModelEloDisplay(model_elo, updated_at)
                        except Exception as exc:
                            self.setStatus(
                                f"elo simulation save failed: {exc}")
                        self.setStatus(
                            f"elo simulation done: model Elo {model_elo}")
                    else:
                        self.setStatus(
                            "elo simulation done: no games completed")
                    self.eloSimActive = False
                    self.eloSimCancelRequested = False

                self.root.after(0, apply_done)

            except Exception as exc:
                def apply_error():
                    self.setStatus(f"elo simulation failed: {exc}")
                    self.eloSimActive = False
                    self.eloSimCancelRequested = False
                self.root.after(0, apply_error)
            finally:
                try:
                    sim_fish.quit()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def cancelMatches(self):
        """Request cancellation of the background match runner."""
        self.matchCancelRequested = True
        self.setStatus("match runner: cancel requested")

    def runMatchesInBackground(self):
        """Run a batch of model vs Stockfish games on a worker thread."""
        if self.matchRunnerActive:
            messagebox.showinfo(
                "Match Runner", "A match run is already active.")
            return
        try:
            numGames = int(self.matchGamesEntry.get())
            maxPlies = int(self.matchMaxPliesEntry.get())
            if numGames <= 0 or maxPlies <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Error", "Games and Max plies must be positive integers.")
            return
        self.matchRunnerActive = True
        self.matchCancelRequested = False
        self.setStatus(f"match runner: starting {numGames} game(s)...")
        modelDepth = int(self.searchDepthVar.get())
        modelThinkMs = int(self.thinkTimeVar.get())
        alternate = bool(self.matchAlternateColorsVar.get())
        verbose = bool(self.matchVerboseVar.get())
        if guiStockfishTimeMs and guiStockfishTimeMs > 0:
            sfLimit = chess.engine.Limit(time=guiStockfishTimeMs / 1000.0)
        else:
            sfLimit = chess.engine.Limit(depth=guiStockfishDepth)
        sfSkill = int(self.sfSkillVar.get())
        sfUseElo = bool(self.sfUseEloVar.get())
        sfElo = int(self.sfEloVar.get())

        def worker():
            matchFish = chess.engine.SimpleEngine.popen_uci(self.stockfishPath)
            try:
                try:
                    matchFish.configure({"Skill Level": sfSkill})
                except Exception:
                    pass
                try:
                    matchFish.configure({"UCI_LimitStrength": sfUseElo})
                except Exception:
                    pass
                if sfUseElo:
                    try:
                        matchFish.configure({"UCI_Elo": sfElo})
                    except Exception:
                        pass
            except Exception:
                pass
            modelWins = 0
            modelLosses = 0
            draws = 0

            def uiAppend(text: str):
                self.root.after(0, lambda: (self.moveList.insert(
                    tk.END, text + "\n"), self.moveList.see(tk.END)))

            def uiStatus(text: str):
                self.root.after(0, lambda: self.setStatus(text))
            try:
                uiAppend("=== Match Runner Started ===")
                if sfUseElo:
                    uiAppend(f"Stockfish: Skill={sfSkill}, Elo≈{sfElo}")
                else:
                    uiAppend(f"Stockfish: Skill={sfSkill} (no Elo limit)")
                uiAppend(
                    f"Model: depth<= {modelDepth}, think={modelThinkMs}ms")
                uiAppend(
                    f"Games={numGames}, Max plies/game={maxPlies}, Alternate colors={alternate}")
                uiAppend("")
                for gameIndex in range(1, numGames + 1):
                    if self.matchCancelRequested:
                        uiStatus("match runner: cancelled")
                        uiAppend("=== Match Runner Cancelled ===")
                        break
                    board = chess.Board()
                    if alternate:
                        modelIsWhite = (gameIndex % 2 == 1)
                    else:
                        modelIsWhite = bool(self.modelPlaysWhiteVar.get())
                    uiStatus(
                        f"match runner: game {gameIndex}/{numGames} (model={'White' if modelIsWhite else 'Black'})")
                    pliesPlayed = 0
                    while not board.is_game_over() and pliesPlayed < maxPlies:
                        if self.matchCancelRequested:
                            break
                        modelToMove = (board.turn == chess.WHITE) if modelIsWhite else (
                            board.turn == chess.BLACK)
                        if modelToMove:
                            result = self.engine.searchIterativeDeepening(
                                board, maxDepth=modelDepth, timeLimitMs=modelThinkMs)
                            move = result.bestMove
                            if move is None:
                                break
                            board.push(move)
                        else:
                            sfResult = matchFish.play(board, sfLimit)
                            if sfResult.move is None:
                                break
                            board.push(sfResult.move)
                        pliesPlayed += 1
                    if board.is_game_over():
                        res = board.result()
                    else:
                        res = "1/2-1/2"
                    if res == "1/2-1/2":
                        draws += 1
                        outcomeText = "Draw"
                    elif res == "1-0":
                        if modelIsWhite:
                            modelWins += 1
                            outcomeText = "Model win"
                        else:
                            modelLosses += 1
                            outcomeText = "Model loss"
                    elif res == "0-1":
                        if not modelIsWhite:
                            modelWins += 1
                            outcomeText = "Model win"
                        else:
                            modelLosses += 1
                            outcomeText = "Model loss"
                    else:
                        draws += 1
                        outcomeText = f"Unknown result treated as draw: {res}"
                    if verbose:
                        uiAppend(
                            f"Game {gameIndex}/{numGames}: {outcomeText} (result={res}, plies={pliesPlayed})")
                    if (gameIndex % max(1, numGames // 10)) == 0 or gameIndex == numGames:
                        uiAppend(
                            f"[progress] {gameIndex}/{numGames} | W={modelWins} L={modelLosses} D={draws}")
                uiAppend("")
                uiAppend("=== Match Runner Finished ===")
                uiAppend(f"Final: W={modelWins} L={modelLosses} D={draws}")
                uiStatus(
                    f"match runner done: W={modelWins} L={modelLosses} D={draws}")
            except Exception as exc:
                uiAppend(f"=== Match Runner Error === {exc}")
                uiStatus(f"match runner failed: {exc}")
            finally:
                try:
                    matchFish.quit()
                except Exception:
                    pass
                self.matchRunnerActive = False
                self.matchCancelRequested = False
        threading.Thread(target=worker, daemon=True).start()

    def onToggleAutoplay(self):
        """Start or stop autoplay depending on the current mode."""
        if self.mode == "engineVsEngine":
            if self.autoPlayVar.get():
                self._autoPlayStep()
            return
        if self.mode == "modelVsStockfish" and self.autoPlayVar.get():
            self._autoPlayModelVsStockfishStep()

    def _autoPlayStep(self):
        """Single autoplay step for engine-vs-engine games."""
        if self.mode != "engineVsEngine":
            return
        if not self.autoPlayVar.get():
            return
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", f"Result: {self.board.result()}")
            return
        depth = int(self.searchDepthVar.get())
        thinkMs = int(self.thinkTimeVar.get())

        def worker():
            boardCopy = self.board.copy(stack=False)
            result = self.engine.searchIterativeDeepening(
                boardCopy, maxDepth=depth, timeLimitMs=thinkMs
            )

            def applyAndContinue():
                self._applyEngineResult(
                    result.bestMove, result.nodes, result.bestScore, result.depth
                )
                self.root.after(120, self._autoPlayStep)

            self.root.after(0, applyAndContinue)
        threading.Thread(target=worker, daemon=True).start()

    def _maybeTriggerAIMove(self):
        """Trigger the next AI move if the mode and state allow it."""
        if self.board.is_game_over():
            return
        if self.mode == "engineVsEngine":
            if self.autoPlayVar.get():
                self._autoPlayStep()
            return
        if self.mode == "modelVsStockfish":
            if self.autoPlayVar.get():
                self._autoPlayModelVsStockfishStep()
            else:
                if self._isStockfishTurn():
                    self._stockfishMoveNow()

    def _isStockfishTurn(self) -> bool:
        """Return True when Stockfish should move in model-vs-Stockfish."""
        modelPlaysWhite = self.modelPlaysWhiteVar.get()
        modelToMove = (self.board.turn == chess.WHITE) if modelPlaysWhite else (
            self.board.turn == chess.BLACK)
        return not modelToMove

    def _autoPlayModelVsStockfishStep(self):
        """Single autoplay step for model-vs-Stockfish games."""
        if self.mode != "modelVsStockfish":
            return
        if not self.autoPlayVar.get():
            return
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", f"Result: {self.board.result()}")
            return
        if self._isStockfishTurn():
            self._stockfishMoveNow(continueAutoplay=True)
        else:
            self._modelMoveNow(continueAutoplay=True)

    def trainInBackground(self):
        """Train a new model on a worker thread and update the engine."""
        if self.trainingActive:
            messagebox.showinfo("Training", "Training is already running.")
            return

        try:
            sampleCount = int(self.sampleEntry.get())
            stockfishDepth = int(self.depthEntry.get())
        except ValueError:
            messagebox.showerror("Error", "Training fields must be integers.")
            return

        self.trainingActive = True
        self.trainingStopRequested = False
        self.stopCollectButton.config(state=tk.NORMAL)

        self.setStatus("training: starting...")

        def worker():
            try:
                print(
                    f"[train] starting: samples={sampleCount}, depth={stockfishDepth}", flush=True)

                model = trainEvalModel(
                    stockfishPath=self.stockfishPath,
                    sampleCount=sampleCount,
                    stockfishDepth=stockfishDepth,
                    shouldStop=lambda: self.trainingStopRequested,
                    reduceDupes=True,
                )

                def applyModel():
                    self.model = model
                    self.engine = AlphaBetaEngine(
                        self.model, openingDbPath=r"D:\ChessEngine\opening_book.sqlite")
                    self.setStatus("training complete (model updated)")
                    self.updateEvalBarsAsync()
                    self.trainingActive = False
                    self.trainingStopRequested = False
                    self.stopCollectButton.config(state=tk.DISABLED)

                self.root.after(0, applyModel)

            except Exception as exc:
                errorText = str(exc)

                def showError(message=errorText):
                    self.setStatus("training failed (see error)")
                    self.trainingActive = False
                    self.trainingStopRequested = False
                    self.stopCollectButton.config(state=tk.DISABLED)
                    messagebox.showerror("Training Error", message)

                self.root.after(0, showError)

        threading.Thread(target=worker, daemon=True).start()

    def saveWeights(self):
        """Persist the current model weights to disk."""
        saveModel(self.model, weightsPath)
        self.setStatus("weights saved")

    def loadWeights(self):
        """Load model weights from disk and rebuild the engine."""
        model = loadModel(weightsPath)
        if model is None:
            messagebox.showerror("Error", "No weights.json found yet.")
            return
        self.model = model
        self.engine = AlphaBetaEngine(
            self.model, openingDbPath=r"D:\ChessEngine\opening_book.sqlite")
        self.setStatus("weights loaded (engine updated)")
        self.updateEvalBarsAsync()
