import pytest

from src.core.vision.hands.detector import HandDetector
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
