"""
Result classes for hand detection in the vision system.
"""

from enum import Enum
from typing import Dict, Tuple

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions.hands import HandLandmark as HL

POINT_2D = Tuple[float, float]


class Finger(str, Enum):
    """Enum representing fingers of left and right hands."""

    LEFT_THUMB = "LEFT_THUMB"
    LEFT_INDEX_FINGER = "LEFT_INDEX"
    LEFT_MIDDLE_FINGER = "LEFT_MIDDLE"
    LEFT_RING_FINGER = "LEFT_RING"
    LEFT_PINKY = "LEFT_PINKY"
    RIGHT_THUMB = "RIGHT_THUMB"
    RIGHT_INDEX_FINGER = "RIGHT_INDEX"
    RIGHT_MIDDLE_FINGER = "RIGHT_MIDDLE"
    RIGHT_RING_FINGER = "RIGHT_RING"
    RIGHT_PINKY = "RIGHT_PINKY"


class HandDetectionResult:
    """
    Encapsulates the result of mediapipe hand detection.

    Attributes:
        label (str): The detected hand label ("Left" or "Right").
        score (float): Confidence score of the hand detection.
        landmarks (NormalizedLandmarkList): Raw landmark points detected by MediaPipe.
        image_shape (Tuple[int, int]): Image height and width for scaling landmarks.
        bbox (Tuple[POINT_2D, POINT_2D]): Bounding box coordinates (top-left, bottom-right).
        fingers (Dict[Finger, bool]): Dictionary indicating which fingers are upright.
        finger_count (int): Number of fingers detected as upright.
    """

    def __init__(
        self,
        label: str,
        score: float,
        landmarks: NormalizedLandmarkList,
        image_shape: Tuple[int, int],
    ):
        self.label = label
        self.score = score
        self.landmarks = landmarks
        self.image_shape = image_shape

        self.bbox = self.get_bounding_box()
        self.fingers = self.get_finger_orientation()
        self.finger_count = sum(self.fingers.values())

    def get_bounding_box(self) -> Tuple[POINT_2D, POINT_2D]:
        """Returns pixel coordinates of outer edges of the detected hand."""
        x_coords = [lmark.x for lmark in self.landmarks.landmark]
        y_coords = [lmark.y for lmark in self.landmarks.landmark]
        h, w = self.image_shape
        xmin, xmax = min(x_coords) * w, max(x_coords) * w
        ymin, ymax = min(y_coords) * h, max(y_coords) * h
        return (xmin, ymin), (xmax, ymax)

    def get_finger_orientation(self) -> Dict[Finger, bool]:
        """
        Determines which fingers are upright based on relative position
        of a finger's tip to its joints.

        Suffixes:
            - _TIP: Tip of the finger
            - _PIP: Proximal interphalangeal joint
            - _MCP: Metacarpophalangeal joint.
        """

        def get_x(lmark):
            return self.landmarks.landmark[lmark].x

        def get_y(lmark):
            return self.landmarks.landmark[lmark].y

        is_index_up = get_y(HL.INDEX_FINGER_TIP) < get_y(HL.INDEX_FINGER_PIP)
        is_middle_up = get_y(HL.MIDDLE_FINGER_TIP) < get_y(HL.MIDDLE_FINGER_PIP)
        is_ring_up = get_y(HL.RING_FINGER_TIP) < get_y(HL.RING_FINGER_PIP)
        is_pinky_up = get_y(HL.PINKY_TIP) < get_y(HL.PINKY_PIP)

        # Use x-coordinates to determine if thumb is "up"
        if self.label == "Right":
            if get_x(HL.THUMB_TIP) > get_x(HL.PINKY_MCP):
                # when right palm is facing away from camera
                is_thumb_up = get_x(HL.THUMB_TIP) > get_x(HL.THUMB_MCP)
            else:
                # when right palm is facing toward the camera
                is_thumb_up = get_x(HL.THUMB_TIP) < get_x(HL.THUMB_MCP)
        elif self.label == "Left":
            if get_x(HL.THUMB_TIP) < get_x(HL.PINKY_MCP):
                # when left palm is facing away from camera
                is_thumb_up = get_x(HL.THUMB_TIP) < get_x(HL.THUMB_MCP)
            else:
                # when left palm is facing toward the camera
                is_thumb_up = get_x(HL.THUMB_TIP) > get_x(HL.THUMB_MCP)

        # Prefix with the hand label (e.g., "RIGHT_THUMB", "LEFT_INDEX")
        prefix_ = self.label.upper() + "_"
        return {
            Finger(f"{prefix_}THUMB"): is_thumb_up,
            Finger(f"{prefix_}INDEX"): is_index_up,
            Finger(f"{prefix_}MIDDLE"): is_middle_up,
            Finger(f"{prefix_}RING"): is_ring_up,
            Finger(f"{prefix_}PINKY"): is_pinky_up,
        }
