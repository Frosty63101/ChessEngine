"""Application entry point for the chess GUI."""

import tkinter as tk
from tkinter import messagebox

from config import stockfishPath, weightsPath
from gui import ChessGuiApp
from model import loadModel


def main():
    try:
        model = loadModel(weightsPath)
        root = tk.Tk()
        app = ChessGuiApp(root, stockfishPath=stockfishPath,
                          initialModel=model)
        root.mainloop()
    except Exception as e:
        # Surface startup failures to the user instead of a silent crash.
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    main()
