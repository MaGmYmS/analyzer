import numpy as np


class Frame:
    predicted_class: int
    confidence: float
    coords: np.ndarray

    def __init__(self, predicted_class, confidence, coords):
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.coords = coords
