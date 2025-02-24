"""
Tests for Hand detector.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmark,
    NormalizedLandmarkList,
)

from src.core.vision.artist import Artist, DrawingConfig
from src.core.vision.hands.detector import HandDetector, HandDetectorArtist
from src.core.vision.hands.result import Finger, HandDetectionResult
from tests.core.vision.utils import load_image_rgb

# Test cases: (image path, expected upright fingers)
TEST_CASES = [
    pytest.param(
        "./tests/assets/right-hand-volume-sign.jpg",
        {
            Finger.RIGHT_THUMB,
            Finger.RIGHT_MIDDLE_FINGER,
            Finger.RIGHT_RING_FINGER,
            Finger.RIGHT_PINKY,
        },
        id="Volume Sign",
    ),
]


@pytest.fixture
def hand_detector():
    """Fixture to provide an instance of HandDetector."""
    return HandDetector(static_image_mode=True)


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
def hand_detector_artist():
    return HandDetectorArtist(draw_config=DrawingConfig())


@pytest.mark.parametrize("image_path, expected_fingers", TEST_CASES)
def test_hand_detection(hand_detector, image_path, expected_fingers):
    """Assert HandDetector correctly identifies hands."""
    image_rgb = load_image_rgb(image_path)
    detected_hands = hand_detector.detect(image_rgb)

    assert len(detected_hands) > 0

    for hand in detected_hands:
        assert isinstance(hand, HandDetectionResult)
        assert hand.label in ("Left", "Right")
        assert hand.fingers
        assert all(isinstance(finger, Finger) for finger in hand.fingers.keys())
        assert hand.landmarks
        assert hand.image_shape
        assert hand.bbox

    detected_fingers = {
        finger
        for hand in detected_hands
        for finger, is_up in hand.fingers.items()
        if is_up
    }
    assert detected_fingers == expected_fingers


@patch.object(HandDetector, "detect", return_value=[MagicMock()])
@patch.object(Artist, "draw_hand")
def test_hand_detector_artist(
    mock_draw_hand,
    mock_detect,
    hand_detector_artist,
    test_image,
    fake_hand_result,
):
    """Assert HandDetectorArtist calls detect() and passes result to draw_hand()."""
    mock_detect.return_value = [fake_hand_result]

    results = hand_detector_artist.detect(test_image)

    mock_detect.assert_called_once_with(test_image)
    mock_draw_hand.assert_called_once_with(test_image, [fake_hand_result])
    assert results == [fake_hand_result]
