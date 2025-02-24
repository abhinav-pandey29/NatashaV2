from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for vision-based detectors."""

    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> List[Any]:
        """
        Process an RGB image and return a list of detections.
        Each detection could include bounding boxes, landmarks, scores, etc.
        """
        pass
