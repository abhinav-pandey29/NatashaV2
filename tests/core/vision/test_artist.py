from unittest.mock import patch

import numpy as np
import pytest
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmark,
    NormalizedLandmarkList,
)

from src.core.vision.artist import Artist, DrawingConfig
from src.core.vision.hands.result import HandDetectionResult


@pytest.fixture
def test_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fake_hand_result(test_image):
    landmarks = NormalizedLandmarkList(
        # 21 hand landmarks in center position
        landmark=[NormalizedLandmark(x=0.5, y=0.5, z=0) for _ in range(21)]
    )
    return HandDetectionResult(
        label="Right",
        score=0.9,
        landmarks=landmarks,
        image_shape=test_image.shape[:2],
    )


@pytest.fixture
def artist():
    return Artist(DrawingConfig())


@patch("cv2.putText")
@patch("cv2.rectangle")
@patch("mediapipe.solutions.drawing_utils.draw_landmarks")
def test_draw_hand(
    mock_draw_landmarks,
    mock_rectangle,
    mock_put_text,
    artist,
    test_image,
    fake_hand_result,
):
    """Ensures draw_hand() calls OpenCV drawing functions correctly."""
    artist.draw_hand(test_image, [fake_hand_result])

    assert mock_put_text.call_count >= 1
    mock_put_text.assert_any_call(
        test_image,
        "Right (0.90)",
        (320, 240),
        artist.config.font_type,
        artist.config.font_scale,
        artist.config.text_color,
        artist.config.text_thickness,
        artist.config.line_type,
    )

    assert mock_rectangle.call_count == 1
    mock_draw_landmarks.assert_called_once_with(
        test_image,
        fake_hand_result.landmarks,
        artist.config.hand_connections,
        artist.config.joint_drawing_spec,
        artist.config.connection_drawing_spec,
    )
