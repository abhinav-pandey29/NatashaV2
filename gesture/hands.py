from enum import Enum
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.python.solutions.hands import HandLandmark as HL
from mediapipe.python.solutions.hands import Hands

from settings import settings


class Finger(str, Enum):

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


class HandDetector:
    """Wrapper for Hand Detector service."""

    detector = Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.6,
    )

    def find_hands(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Returns array of hands from an RGB image."""
        found_hands = []
        hand_results = self.detector.process(image_rgb)
        if hand_results.multi_handedness:
            for idx, handedness in enumerate(hand_results.multi_handedness):
                score = handedness.classification[0].score
                label = handedness.classification[0].label
                landmarks = hand_results.multi_hand_landmarks[idx]
                fingers = self.get_finger_orientation(
                    hand_landmarks=landmarks,
                    hand_label=label,
                )
                found_hands.append(
                    {
                        "label": label,
                        "score": score,
                        "landmarks": landmarks,
                        "fingers": fingers,
                        "finger_count": sum(fingers.values()),
                    }
                )

        return found_hands

    @staticmethod
    # TODO: This only works when the hands are upright, which causes the detector to
    # misinterpret finger orientation when the camera is upside-down.
    def get_finger_orientation(
        hand_landmarks: NormalizedLandmarkList,
        hand_label: str,
    ) -> Dict[Finger, bool]:
        """
        Determines which fingers are upright based on relative position
        of a finger's tip to its joints.
        Suffixes:
            - _TIP: Tip of the finger
            - _PIP: Proximal interphalangeal joint
            - _MCP: Metacarpophalangeal joint.
        """
        get_x = lambda lmark: hand_landmarks.landmark[lmark].x
        get_y = lambda lmark: hand_landmarks.landmark[lmark].y

        is_index_up = get_y(HL.INDEX_FINGER_TIP) < get_y(HL.INDEX_FINGER_PIP)
        is_middle_up = get_y(HL.MIDDLE_FINGER_TIP) < get_y(HL.MIDDLE_FINGER_PIP)
        is_ring_up = get_y(HL.RING_FINGER_TIP) < get_y(HL.RING_FINGER_PIP)
        is_pinky_up = get_y(HL.PINKY_TIP) < get_y(HL.PINKY_PIP)
        # Use x-coordinates to determine if thumb is "up", since it opens and
        # closes horizontally.
        if hand_label == "Right":
            if get_x(HL.THUMB_TIP) > get_x(HL.PINKY_MCP):
                # when right palm is facing away from camera
                is_thumb_up = get_x(HL.THUMB_TIP) > get_x(HL.THUMB_MCP)
            else:
                # when right palm is facing toward the camera
                is_thumb_up = get_x(HL.THUMB_TIP) < get_x(HL.THUMB_MCP)
        elif hand_label == "Left":
            if get_x(HL.THUMB_TIP) < get_x(HL.PINKY_MCP):
                # when left palm is facing away from camera
                is_thumb_up = get_x(HL.THUMB_TIP) < get_x(HL.THUMB_MCP)
            else:
                # when left palm is facing toward the camera
                is_thumb_up = get_x(HL.THUMB_TIP) > get_x(HL.THUMB_MCP)

        prefix_ = hand_label.upper() + "_"
        return {
            f"{prefix_}THUMB": is_thumb_up,
            f"{prefix_}INDEX": is_index_up,
            f"{prefix_}MIDDLE": is_middle_up,
            f"{prefix_}RING": is_ring_up,
            f"{prefix_}PINKY": is_pinky_up,
        }


class HandDetectorArtist(HandDetector):
    """Hand detector service with additional drawing features"""

    def __init__(
        self,
        font_type=None,
        text_color=None,
        line_type=None,
        joint_drawing_spec=None,
        connection_drawing_spec=None,
    ):
        self._font_type = font_type
        self._text_color = text_color
        self._line_type = line_type
        self._joint_drawing_spec = joint_drawing_spec
        self._connection_drawing_spec = connection_drawing_spec

    @property
    def font_type(self):
        return self._font_type or settings.CV2_FONT_TYPE

    @property
    def text_color(self):
        return self._text_color or settings.CV2_TEXT_COLOR

    @property
    def line_type(self):
        return self._line_type or settings.CV2_LINE_TYPE

    @property
    def joint_drawing_spec(self):
        return self._joint_drawing_spec or settings.HAND_JOINT_SPEC

    @property
    def connection_drawing_spec(self):
        return self._connection_drawing_spec or settings.HAND_CONNECTION_SPEC

    def set_font_type(self, font_type):
        self._font_type = font_type

    def set_text_color(self, text_color):
        self._text_color = text_color

    def set_line_type(self, line_type):
        self._line_type = line_type

    def set_joint_drawing_spec(self, spec):
        self._joint_drawing_spec = spec

    def set_connection_drawing_spec(self, spec):
        self._connection_drawing_spec = spec

    def reset_artist(self):
        self.set_font_type(None)
        self.set_text_color(None)
        self.set_line_type(None)
        self.set_joint_drawing_spec(None)
        self.set_connection_drawing_spec(None)

    def find_hands(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns array of hands from an RGB image and
        annotates image with relevant info.
        """
        found_hands = super().find_hands(image_rgb)
        line_spacing = 1.5
        uv_top_left = np.array((5, 40), dtype=float)
        for hand in found_hands:
            # Draw hand landmarks
            drawing_utils.draw_landmarks(
                image_rgb,
                hand["landmarks"],
                HAND_CONNECTIONS,
                self.joint_drawing_spec,
                self.connection_drawing_spec,
            )
            # Draw hand label (Left, Right) and score
            label_target = hand["landmarks"].landmark[HL.WRIST]
            label_pos = np.array([label_target.x, label_target.y])  # Ratios
            label_pos = (label_pos * [640, 480]).astype(int)  # Pixels
            cv2.putText(
                image_rgb,
                f"{hand['label']} ({hand['score']:.2f})",
                label_pos,
                self.font_type,
                0.5,
                self.text_color,
                1,
                self.line_type,
            )
            # Draw finger labels
            gesture = [k for k, v in hand["fingers"].items() if v]
            if gesture:
                line = f"{hand['label']}: {' '.join(gesture)}"
                (w, h), _ = cv2.getTextSize(
                    text=line,
                    fontFace=self.font_type,
                    fontScale=0.5,
                    thickness=1,
                )
                uv_bottom_left_i = uv_top_left + [0, h]
                org = tuple(uv_bottom_left_i.astype(int))
                cv2.putText(
                    image_rgb,
                    text=line,
                    org=org,
                    fontFace=self.font_type,
                    fontScale=0.35,
                    color=self.text_color,
                    thickness=1,
                    lineType=self.line_type,
                )
                uv_top_left += [0, h * line_spacing]
        # Draw finger count
        total_fingers = sum(hand["finger_count"] for hand in found_hands)
        label_pos = (5, 25)
        cv2.putText(
            image_rgb,
            f"Total Finger: {total_fingers}",
            label_pos,
            self.font_type,
            0.5,
            self.text_color,
            1,
            self.line_type,
        )

        return found_hands


def get_hand_detector(draw: bool = True) -> Union[HandDetector, HandDetectorArtist]:
    """Factory for Hand Detector service."""
    return HandDetectorArtist() if draw else HandDetector()
