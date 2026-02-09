"""Lightweight evaluation model utilities for chess positions."""

import json
from dataclasses import dataclass

import chess
import numpy as np

from features import boardToFeatureVector, featureNames


@dataclass
class EvalModel:
    """Linear evaluation model backed by a weight vector."""

    weights: np.ndarray

    def predict(self, board: chess.Board) -> float:
        """Score a board position using the feature vector dot product."""
        featureVector = boardToFeatureVector(board)
        return float(np.dot(self.weights, featureVector))

    def toJson(self) -> dict:
        """Serialize model weights and feature names for persistence."""
        return {
            "weights": self.weights.tolist(),
            "featureNames": featureNames(),
        }

    @staticmethod
    def fromJson(obj: dict) -> "EvalModel":
        """Create a model from serialized JSON data."""
        return EvalModel(weights=np.array(obj["weights"], dtype=np.float64))


def saveModel(model: EvalModel, weightsPath: str):
    """Write model weights and metadata to a JSON file."""
    with open(weightsPath, "w", encoding="utf-8") as fileHandle:
        json.dump(model.toJson(), fileHandle, indent=2)


def loadModel(weightsPath: str) -> EvalModel | None:
    """Load a model from JSON, returning None when the file is missing."""
    try:
        with open(weightsPath, "r", encoding="utf-8") as fileHandle:
            obj = json.load(fileHandle)
        return EvalModel.fromJson(obj)
    except FileNotFoundError:
        return None
