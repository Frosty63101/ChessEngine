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
        """
        Score a board position using the feature vector dot product.

        This function expects self.weights and the feature vector to have the same length.
        If they don't, it raises a clear error telling you to regenerate/migrate weights.json.
        """
        featureVector = boardToFeatureVector(board)

        if self.weights.shape[0] != featureVector.shape[0]:
            raise ValueError(
                f"EvalModel weight length ({self.weights.shape[0]}) does not match "
                f"feature vector length ({featureVector.shape[0]}). "
                "Your weights.json is out of date for the current features. "
                "Delete weights.json or let loadModel() migrate it."
            )

        return float(np.dot(self.weights, featureVector))

    def toJson(self) -> dict:
        """Serialize model weights and feature names for persistence."""
        return {
            "weights": self.weights.tolist(),
            "featureNames": featureNames(),
        }

    @staticmethod
    def fromJson(obj: dict) -> "EvalModel":
        """
        Create a model from serialized JSON data.

        If the stored weights/featureNames don't match the current feature set,
        we migrate by feature name (best-effort).
        """
        currentNames = featureNames()
        currentCount = len(currentNames)

        rawWeights = obj.get("weights", None)
        if not isinstance(rawWeights, list):
            # Corrupt or missing weights: fall back to zeros.
            return EvalModel(weights=np.zeros(currentCount, dtype=np.float64))

        storedNames = obj.get("featureNames", None)

        # If the file had feature names, migrate by name.
        if isinstance(storedNames, list) and all(isinstance(x, str) for x in storedNames):
            # Build name->weight map from the old file
            oldCount = min(len(storedNames), len(rawWeights))
            oldMap = {storedNames[i]: float(rawWeights[i])
                      for i in range(oldCount)}

            # Construct new weight vector aligned to currentNames
            newWeights = np.zeros(currentCount, dtype=np.float64)
            migrated = 0

            for i, name in enumerate(currentNames):
                if name in oldMap:
                    newWeights[i] = oldMap[name]
                    migrated += 1

            print(
                f"[model] migrated weights by feature name: {migrated}/{currentCount} matched. "
                f"(oldFeatures={len(storedNames)}, newFeatures={currentCount})",
                flush=True,
            )
            return EvalModel(weights=newWeights)

        # Otherwise: no names stored. Only accept if length matches exactly.
        if len(rawWeights) == currentCount:
            return EvalModel(weights=np.array(rawWeights, dtype=np.float64))

        print(
            f"[model] WARNING: weights.json has {len(rawWeights)} weights but current features have {currentCount}. "
            "No featureNames found in file, so migration is not possible. Resetting to zeros.",
            flush=True,
        )
        return EvalModel(weights=np.zeros(currentCount, dtype=np.float64))


def saveModel(model: EvalModel, weightsPath: str):
    """Write model weights and metadata to a JSON file."""
    with open(weightsPath, "w", encoding="utf-8") as fileHandle:
        json.dump(model.toJson(), fileHandle, indent=2)


def loadModel(weightsPath: str) -> EvalModel | None:
    """
    Load a model from JSON, returning None when the file is missing.

    Important behavior:
      - If weights.json is from an older feature set, we migrate it safely.
      - If migration occurs, we immediately save the upgraded file back to disk.
    """
    try:
        with open(weightsPath, "r", encoding="utf-8") as fileHandle:
            obj = json.load(fileHandle)

        model = EvalModel.fromJson(obj)

        # If the loaded file differs from current feature set, rewrite it so GUI/train match next boot.
        # We detect that by comparing stored featureNames to current.
        storedNames = obj.get("featureNames", None)
        currentNames = featureNames()
        needsRewrite = not (isinstance(storedNames, list)
                            and storedNames == currentNames)

        if needsRewrite:
            saveModel(model, weightsPath)
            print(
                f"[model] upgraded weights file in-place: {weightsPath}", flush=True)

        return model

    except FileNotFoundError:
        return None
    except Exception as exc:
        print(
            f"[model] ERROR loading weights from {weightsPath}: {exc}", flush=True)
        return None
