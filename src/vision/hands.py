from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.hands import HAND_CONNECTIONS, Hands
from mediapipe.python.solutions.hands import HandLandmark as HL

from config.settings import settings

POINT_2D = Tuple[float, float]


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

    def __init__(self):
        self.detector = Hands(
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
                bbox = self.get_bounding_box(landmarks, *image_rgb.shape)
                found_hands.append(
                    {
                        "label": label,
                        "score": score,
                        "landmarks": landmarks,
                        "fingers": fingers,
                        "finger_count": sum(fingers.values()),
                        "bbox": bbox,
                    }
                )

        return found_hands

    @staticmethod
    def get_bounding_box(
        hand_landmarks: NormalizedLandmarkList,
        *img_shape,
    ) -> Tuple[POINT_2D, POINT_2D]:
        """Returns pixel coordinates of outer edges the hands."""
        x_coords = [lmark.x for lmark in hand_landmarks.landmark]
        y_coords = [lmark.y for lmark in hand_landmarks.landmark]
        h, w, *_ = img_shape
        xmin, xmax = min(x_coords) * w, max(x_coords) * w
        ymin, ymax = min(y_coords) * h, max(y_coords) * h
        return (xmin, ymin), (xmax, ymax)

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

        def get_x(lmark):
            return hand_landmarks.landmark[lmark].x

        def get_y(lmark):
            return hand_landmarks.landmark[lmark].y

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
            Finger(f"{prefix_}THUMB"): is_thumb_up,
            Finger(f"{prefix_}INDEX"): is_index_up,
            Finger(f"{prefix_}MIDDLE"): is_middle_up,
            Finger(f"{prefix_}RING"): is_ring_up,
            Finger(f"{prefix_}PINKY"): is_pinky_up,
        }


class HandDetectorArtist(HandDetector):
    """Hand detector service with additional drawing features"""

    def __init__(self, **kwargs):
        super().__init__()
        self.artist_settings = {
            "font_type": settings.CV2_FONT_TYPE,
            "text_color": settings.CV2_TEXT_COLOR,
            "line_type": settings.CV2_LINE_TYPE,
            "joint_drawing_spec": settings.HAND_JOINT_SPEC,
            "connection_drawing_spec": settings.HAND_CONNECTION_SPEC,
            "bbox_padding": settings.BBOX_PADDING,
            **kwargs,
        }

    def update_artist_settings(self, **kwargs):
        """Update drawing settings."""
        self.artist_settings.update(kwargs)

    def reset_artist(self):
        """Reset drawing settings to default."""
        self.update_artist_settings(
            font_type=settings.CV2_FONT_TYPE,
            text_color=settings.CV2_TEXT_COLOR,
            line_type=settings.CV2_LINE_TYPE,
            joint_drawing_spec=settings.HAND_JOINT_SPEC,
            connection_drawing_spec=settings.HAND_CONNECTION_SPEC,
            bbox_padding=settings.BBOX_PADDING,
        )

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
                self.artist_settings["joint_drawing_spec"],
                self.artist_settings["connection_drawing_spec"],
            )
            # Draw hand label (Left, Right) and score
            label_target = hand["landmarks"].landmark[HL.WRIST]
            label_pos = np.array([label_target.x, label_target.y])  # Ratios
            label_pos = (label_pos * [640, 480]).astype(int)  # Pixels
            cv2.putText(
                image_rgb,
                f"{hand['label']} ({hand['score']:.2f})",
                label_pos,
                self.artist_settings["font_type"],
                0.5,
                self.artist_settings["text_color"],
                1,
                self.artist_settings["line_type"],
            )
            # Draw bounding box
            pt1, pt2 = hand["bbox"]
            pt1 = tuple(int(x - self.artist_settings["bbox_padding"]) for x in pt1)
            pt2 = tuple(int(x + self.artist_settings["bbox_padding"]) for x in pt2)
            cv2.rectangle(image_rgb, pt1, pt2, (200, 123, 90), 2)
            # Draw finger labels
            gesture = [k for k, v in hand["fingers"].items() if v]
            if gesture:
                line = f"{hand['label']}: {' '.join(gesture)}"
                (w, h), _ = cv2.getTextSize(
                    text=line,
                    fontFace=self.artist_settings["font_type"],
                    fontScale=0.5,
                    thickness=1,
                )
                uv_bottom_left_i = uv_top_left + [0, h]
                org = tuple(uv_bottom_left_i.astype(int))
                cv2.putText(
                    image_rgb,
                    text=line,
                    org=org,
                    fontFace=self.artist_settings["font_type"],
                    fontScale=0.35,
                    color=self.artist_settings["text_color"],
                    thickness=1,
                    lineType=self.artist_settings["line_type"],
                )
                uv_top_left += [0, h * line_spacing]
        # Draw finger count
        total_fingers = sum(hand["finger_count"] for hand in found_hands)
        label_pos = (5, 25)
        cv2.putText(
            image_rgb,
            f"Total Finger: {total_fingers}",
            label_pos,
            fontFace=self.artist_settings["font_type"],
            fontScale=0.5,
            color=self.artist_settings["text_color"],
            thickness=1,
            lineType=self.artist_settings["line_type"],
        )

        return found_hands


def get_hand_detector(draw: bool = True) -> Union[HandDetector, HandDetectorArtist]:
    """Factory for Hand Detector service."""
    return HandDetectorArtist() if draw else HandDetector()
