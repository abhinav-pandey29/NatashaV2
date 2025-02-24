from typing import List, Optional, Union

import numpy as np
from mediapipe.python.solutions.hands import Hands

from src.core.vision.artist import Artist, DrawingConfig
from src.core.vision.base_detector import BaseDetector

from .result import HandDetectionResult


class HandDetector(BaseDetector):
    """Wrapper for Hand Detector using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.6,
    ):
        self.detector = Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image_rgb: np.ndarray) -> List[HandDetectionResult]:
        """Returns array of hands from an RGB image."""
        results = []
        hand_results = self.detector.process(image_rgb)
        if not hand_results.multi_handedness:
            return results

        for idx, handedness in enumerate(hand_results.multi_handedness):
            score = handedness.classification[0].score
            label = handedness.classification[0].label
            landmarks = hand_results.multi_hand_landmarks[idx]

            results.append(
                HandDetectionResult(
                    label=label,
                    score=score,
                    landmarks=landmarks,
                    image_shape=image_rgb.shape[:2],  # (height, width),
                )
            )

        return results


class HandDetectorArtist(HandDetector):
    """Hand detector service with additional visualization features."""

    def __init__(self, draw_config: Optional[DrawingConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.artist = Artist(draw_config)

    def detect(self, image_rgb: np.ndarray) -> List[HandDetectionResult]:
        """Returns array of hands from an RGB image and draws landmarks."""
        results = super().detect(image_rgb)
        self.artist.draw_hand(image_rgb, results)
        return results


def get_hand_detector(draw: bool = True) -> Union[HandDetector, HandDetectorArtist]:
    """Factory for Hand Detector service."""
    return HandDetectorArtist() if draw else HandDetector()
