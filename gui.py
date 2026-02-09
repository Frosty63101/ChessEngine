"""Tkinter GUI for playing and training the chess evaluation model."""

import os
import threading
import tkinter as tk
from tkinter import messagebox

import chess
import chess.engine
from PIL import Image, ImageTk

from config import (defaultSearchDepth, defaultThinkTimeMs, guiStockfishDepth,
                    guiStockfishTimeMs, imageDir, weightsPath)
from engine import AlphaBetaEngine
from model import EvalModel, loadModel, saveModel
from trainer import trainEvalModel


class EvalBar(tk.Canvas):
    """Vertical evaluation bar that renders a material/score balance."""

    def __init__(self, parent, width=30, height=220, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.width = width
        self.height = height
        self.maxAbs = 10.0
        self.evalValue = 0.0
        self.draw()

    def setEval(self, evalPawns: float):
        """Update the displayed evaluation, clamped to the bar range."""
        if evalPawns > self.maxAbs:
            evalPawns = self.maxAbs
        if evalPawns < -self.maxAbs:
            evalPawns = -self.maxAbs
        self.evalValue = evalPawns
        self.draw()

    def draw(self):
        """Redraw the bar based on the current evaluation value."""
        self.delete("all")
        self.create_rectangle(0, 0, self.width, self.height,
                              fill="gray20", outline="gray50")
        midY = self.height // 2
        self.create_line(0, midY, self.width, midY, fill="gray70")
        ratio = self.evalValue / self.maxAbs
        balanceY = int(midY - ratio * (self.height // 2))
        self.create_rectangle(0, 0, self.width, balanceY,
                              fill="white", outline="")
        self.create_rectangle(0, balanceY, self.width,
                              self.height, fill="black", outline="")
        self.create_text(self.width // 2, self.height - 10, fill="red",
                         text=f"{self.evalValue:+.2f}", anchor="s")


class ChessGuiApp:
    """Main GUI controller that wires UI, engine, model, and Stockfish."""

    def __init__(self, root: tk.Tk, stockfishPath: str, initialModel: EvalModel | None):
        self.root = root
        self.stockfishPath = stockfishPath
        self.evalThreadRunning = False
        self.model = initialModel if initialModel is not None else self._defaultModel()
        self.engine = AlphaBetaEngine(self.model)
        self.board = chess.Board()
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
        self.matchRunnerActive = False
        self.matchCancelRequested = False
        self.applyStockfishStrengthToEngine(self.fishPlayer)
        self._loadPieceImages()
        self._buildUi()
        self.drawBoard()
        self.updateEvalBarsAsync()

    def _defaultModel(self):
        """Create a fallback material-only model when no weights exist."""
        w = [0.0] * 16
        w[1] = 1.0
        w[2] = 3.2
        w[3] = 3.3
        w[4] = 5.1
        w[5] = 9.2
        return EvalModel(weights=self._toNp(w))

    def _toNp(self, w):
        """Convert a Python list of weights to a NumPy vector."""
        import numpy as np
        return np.array(w, dtype=np.float64)

    def _loadPieceImages(self):
        """Load piece sprites from disk into memory for fast rendering."""
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
        """Construct the Tkinter layout and bind UI actions."""
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
        tk.Spinbox(ctrlFrame, from_=1, to=8, textvariable=self.searchDepthVar,
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
        tk.Checkbutton(
            ctrlFrame,
            text="Model plays White (else Black)",
            variable=self.modelPlaysWhiteVar
        ).grid(row=2, column=3, padx=5, sticky="w")
        sfFrame = tk.LabelFrame(
            rightFrame, text="Stockfish Strength (for playing)")
        sfFrame.pack(fill=tk.X, pady=5)
        tk.Label(sfFrame, text="Skill (0-20):").grid(row=0,
                                                     column=0, sticky="w")
        self.sfSkillScale = tk.Scale(
            sfFrame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.sfSkillVar,
            command=lambda _: self.onStockfishStrengthChanged()
        )
        self.sfSkillScale.grid(row=0, column=1, sticky="we", padx=5)
        sfFrame.grid_columnconfigure(1, weight=1)
        self.sfUseEloCheck = tk.Checkbutton(
            sfFrame, text="Limit by Elo", variable=self.sfUseEloVar,
            command=self.onStockfishStrengthChanged
        )
        self.sfUseEloCheck.grid(row=1, column=0, sticky="w")
        tk.Label(sfFrame, text="Elo:").grid(row=1, column=1, sticky="w")
        self.sfEloScale = tk.Scale(
            sfFrame, from_=800, to=2850, resolution=50, orient=tk.HORIZONTAL,
            variable=self.sfEloVar, command=lambda _: self.onStockfishStrengthChanged()
        )
        self.sfEloScale.grid(
            row=2, column=0, columnspan=2, sticky="we", padx=5)
        tk.Button(sfFrame, text="Apply Strength", command=self.onStockfishStrengthChanged).grid(
            row=3, column=0, pady=4, sticky="w")
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
        tk.Button(ctrlFrame, text="Engine Move Now",
                  command=self.engineMoveNow).grid(row=0, column=3, padx=5)
        tk.Checkbutton(ctrlFrame, text="Auto-play self game", variable=self.autoPlayVar,
                       command=self.onToggleAutoplay).grid(row=1, column=3, padx=5)
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

    def onClose(self):
        """Gracefully shut down engines before closing the window."""
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
        """Clear the move list text box and reset the header."""
        self.moveList.delete("1.0", tk.END)
        self.moveList.insert(tk.END, "Moves:\n")

    def setStatus(self, text: str):
        """Update the status line and flush pending UI updates."""
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
            try:
                boardCopy = chess.Board(fen)
                modelEval = self.model.predict(boardCopy)
                try:
                    if guiStockfishTimeMs and guiStockfishTimeMs > 0:
                        info = self.fish.analyse(boardCopy, chess.engine.Limit(
                            time=guiStockfishTimeMs / 1000.0))
                    else:
                        info = self.fish.analyse(
                            boardCopy, chess.engine.Limit(depth=guiStockfishDepth))
                    score = info["score"].pov(chess.WHITE)
                    if score.is_mate():
                        mateIn = score.mate()
                        sfEval = 0.0 if mateIn is None else (
                            100.0 if mateIn > 0 else -100.0)
                    else:
                        cp = score.score(mate_score=100000)
                        sfEval = 0.0 if cp is None else float(cp) / 100.0
                except Exception:
                    sfEval = 0.0

                def apply():
                    self.modelEvalBar.setEval(modelEval)
                    self.sfEvalBar.setEval(sfEval)
            finally:
                self.evalThreadRunning = False
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
        """Kick off a move for the current side to play."""
        if self.board.is_game_over():
            return
        if self.mode == "modelVsStockfish":
            if self._isStockfishTurn():
                self._stockfishMoveNow()
            else:
                self._modelMoveNow()
            return
        depth = int(self.searchDepthVar.get())
        thinkMs = int(self.thinkTimeVar.get())
        self.setStatus(f"thinking (depth<= {depth}, {thinkMs}ms)...")

        def worker():
            result = self.engine.searchIterativeDeepening(
                self.board, maxDepth=depth, timeLimitMs=thinkMs)
            self.root.after(0, lambda: self._applyEngineResult(
                result.bestMove, result.nodes, result.bestScore, result.depth))
        threading.Thread(target=worker, daemon=True).start()

    def _applyEngineResult(self, move, nodes: int, score: float, depth: int):
        """Apply an engine move result and refresh UI state."""
        if move is None:
            self.setStatus("no move found")
            return
        san = self.board.san(move)
        self.board.push(move)
        self._appendMoveSan(san)
        self.drawBoard()
        self.updateEvalBarsAsync()
        self.setStatus(
            f"played {san} | nodes={nodes} | eval≈{score:.2f} pawns | depth={depth}")
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
            result = self.engine.searchIterativeDeepening(
                self.board, maxDepth=depth, timeLimitMs=thinkMs)

            def applyAndContinue():
                self._applyEngineResult(
                    result.bestMove, result.nodes, result.bestScore, result.depth)
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

    def _stockfishMoveNow(self, continueAutoplay: bool = False):
        """Ask Stockfish to play a move, optionally continuing autoplay."""
        self.applyStockfishStrengthToEngine(self.fishPlayer)
        self.setStatus("Stockfish thinking...")

        def worker():
            try:
                if guiStockfishTimeMs and guiStockfishTimeMs > 0:
                    limit = chess.engine.Limit(
                        time=guiStockfishTimeMs / 1000.0)
                else:
                    limit = chess.engine.Limit(depth=guiStockfishDepth)
                result = self.fishPlayer.play(self.board, limit)
                move = result.move
            except Exception as exc:
                move = None
                errText = str(exc)

            def apply():
                if move is None:
                    self.setStatus(f"Stockfish move failed: {errText}")
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
                    self.root.after(120, self._autoPlayModelVsStockfishStep)
            self.root.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def _modelMoveNow(self, continueAutoplay: bool = False):
        """Ask the model/engine to play a move, optionally continuing autoplay."""
        if self.board.is_game_over():
            return
        depth = int(self.searchDepthVar.get())
        thinkMs = int(self.thinkTimeVar.get())
        self.setStatus(f"model thinking (depth<= {depth}, {thinkMs}ms)...")

        def worker():
            result = self.engine.searchIterativeDeepening(
                self.board, maxDepth=depth, timeLimitMs=thinkMs)
            bestMove = result.bestMove
            nodes = result.nodes
            bestScore = result.bestScore
            usedDepth = result.depth

            def apply():
                if bestMove is None:
                    self.setStatus("model found no move")
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
                    f"model played {san} | nodes={nodes} | eval≈{bestScore:.2f} | depth={usedDepth}")
                if continueAutoplay:
                    self.root.after(120, self._autoPlayModelVsStockfishStep)
                else:
                    self._maybeTriggerAIMove()
            self.root.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def trainInBackground(self):
        """Train a new model on a worker thread and update the engine."""
        try:
            sampleCount = int(self.sampleEntry.get())
            stockfishDepth = int(self.depthEntry.get())
        except ValueError:
            messagebox.showerror("Error", "Training fields must be integers.")
            return
        self.setStatus("training: starting...")

        def worker():
            try:
                print(
                    f"[train] starting: samples={sampleCount}, depth={stockfishDepth}", flush=True)
                model = trainEvalModel(
                    stockfishPath=self.stockfishPath,
                    sampleCount=sampleCount,
                    stockfishDepth=stockfishDepth,
                )

                def applyModel():
                    self.model = model
                    self.engine = AlphaBetaEngine(self.model)
                    self.setStatus("training complete (model updated)")
                    self.updateEvalBarsAsync()
                self.root.after(0, applyModel)
            except Exception as exc:
                error_text = str(exc)

                def showError(message=error_text):
                    self.setStatus("training failed (see error)")
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
        self.engine = AlphaBetaEngine(self.model)
        self.setStatus("weights loaded (engine updated)")
        self.updateEvalBarsAsync()
