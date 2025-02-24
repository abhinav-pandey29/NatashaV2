"""
Visualization utilities for vision core.

This module provides the `Artist` class for drawing landmarks, labels, 
and bounding boxes on images using OpenCV.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.python.solutions.hands import HandLandmark as HL

from src.core.vision.hands.result import HandDetectionResult


@dataclass
class DrawingConfig:
    """Configuration for visualization settings in vision modules."""

    font_type: int = cv2.FONT_HERSHEY_DUPLEX
    text_color: Tuple[int, int, int] = (255, 255, 255)
    line_type: int = cv2.LINE_AA
    bbox_padding: int = 15
    font_scale: float = 0.5
    text_thickness: int = 1
    line_spacing: float = 1.5
    bbox_color: Tuple[int, int, int] = (200, 123, 90)
    label_offset: Tuple[int, int] = (5, 25)

    # Hand landmark drawing specifications
    hand_connections = HAND_CONNECTIONS
    joint_drawing_spec: DrawingSpec = drawing_utils.DrawingSpec(
        color=(255, 69, 0),
        thickness=2,
        circle_radius=2,
    )
    connection_drawing_spec: DrawingSpec = drawing_utils.DrawingSpec(
        color=(255, 69, 0),
        thickness=2,
        circle_radius=2,
    )


class Artist:
    """Handles visualization for detected objects (hands, faces, pose, etc.)."""

    def __init__(self, config: DrawingConfig = None):
        self.config = config or DrawingConfig()

    def draw_hand(
        self, image: np.ndarray, detections: List[HandDetectionResult]
    ) -> None:
        """Draws hand landmarks, bounding boxes, and labels for detected hands."""
        uv_top_left = np.array(self.config.label_offset, dtype=float)
        total_fingers = 0

        for hand in detections:
            # Draw landmarks
            drawing_utils.draw_landmarks(
                image,
                hand.landmarks,
                self.config.hand_connections,
                self.config.joint_drawing_spec,
                self.config.connection_drawing_spec,
            )

            # Draw hand label (Left/Right) and score
            label_target = hand.landmarks.landmark[HL.WRIST]
            label_pos = np.array([label_target.x, label_target.y]) * [
                image.shape[1],
                image.shape[0],
            ]
            cv2.putText(
                image,
                f"{hand.label} ({hand.score:.2f})",
                tuple(label_pos.astype(int)),
                self.config.font_type,
                self.config.font_scale,
                self.config.text_color,
                self.config.text_thickness,
                self.config.line_type,
            )

            # Draw bounding box
            self.draw_bounding_box(image, hand.bbox)

            # Draw detected fingers
            detected_fingers = [k for k, v in hand.fingers.items() if v]
            if detected_fingers:
                line = f"{hand.label}: {' '.join(detected_fingers)}"
                (w, h), _ = cv2.getTextSize(
                    line,
                    self.config.font_type,
                    self.config.font_scale,
                    self.config.text_thickness,
                )
                org = tuple((uv_top_left + [0, h]).astype(int))
                cv2.putText(
                    image,
                    line,
                    org,
                    self.config.font_type,
                    self.config.font_scale * 0.7,  # Adjust scale slightly
                    self.config.text_color,
                    self.config.text_thickness,
                    self.config.line_type,
                )
                uv_top_left += [0, h * self.config.line_spacing]

            total_fingers += hand.finger_count

        # Draw total fingers count
        cv2.putText(
            image,
            f"Total Fingers: {total_fingers}",
            self.config.label_offset,
            self.config.font_type,
            self.config.font_scale,
            self.config.text_color,
            self.config.text_thickness,
            self.config.line_type,
        )

    def draw_bounding_box(
        self, image: np.ndarray, bbox: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> None:
        """Draws a bounding box around the detected hand."""
        pt1, pt2 = bbox
        pt1 = tuple(int(x - self.config.bbox_padding) for x in pt1)
        pt2 = tuple(int(x + self.config.bbox_padding) for x in pt2)
        cv2.rectangle(image, pt1, pt2, self.config.bbox_color, 2)
