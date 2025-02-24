from typing import List

import cv2

from src.commands.gesture_commands import (
    GestureCommand,
    exit_vision_command_factory,
    open_spotify_command_factory,
    play_next_track_command_factory,
    play_prev_track_command_factory,
    set_volume_command_factory,
    shuffle_saved_tracks_command_factory,
)
from src.core.vision.hands.detector import get_hand_detector
from src.integrations.spotify import Spotify


class Vision:
    """Primary class for Natasha's Vision system."""

    def __init__(
        self,
        camera_id: int = 0,
        gesture_commands: List[GestureCommand] = None,
        draw: bool = True,
    ):
        self.camera_id = camera_id
        self.hand_detector = get_hand_detector(draw=draw)
        self.gesture_commands = gesture_commands if gesture_commands is not None else []
        self._draw = draw

    def process_frame(self, cap):
        """Captures and preprocesses a frame from the camera."""
        _, frame = cap.read()
        img_flipped = cv2.flip(frame, 1)  # Flip on horizontal axis
        return cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)

    def detect_gesture_command(self, image_rgb):
        """Detects a gesture command based on hand landmarks."""
        found_hands = self.hand_detector.detect(image_rgb)
        found_gesture = [
            f for hand in found_hands for f, is_up in hand.fingers.items() if is_up
        ]
        command_match = next(
            (
                command
                for command in self.gesture_commands
                if command.gesture.match(found_gesture)
            ),
            None,
        )
        return image_rgb, found_hands, command_match

    def execute_gesture(self, command, img, found_hands, cap, hidden):
        """Executes a detected gesture command."""
        if not hidden:
            cv2.imshow("Hand Tracking", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        command(
            hand_detector=self.hand_detector,
            image=img,
            found_hands=found_hands,
            draw=self._draw,
            cap=cap,
            hidden=hidden,
        )

    def activate(self, hidden: bool = False) -> None:
        """Activates the vision system for detecting and executing gestures."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise IOError("Cannot open webcam :(")

        while cap.isOpened():
            img_rgb = self.process_frame(cap)
            img, found_hands, activated_command = self.detect_gesture_command(img_rgb)

            if activated_command:
                self.execute_gesture(activated_command, img, found_hands, cap, hidden)

            if not hidden:
                cv2.imshow("Hand Tracking", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break


if __name__ == "__main__":
    spotify_client = Spotify()

    spotify_gestures = [
        exit_vision_command_factory(),
        open_spotify_command_factory(),
        shuffle_saved_tracks_command_factory(spotify_client),
        play_next_track_command_factory(spotify_client),
        play_prev_track_command_factory(spotify_client),
        set_volume_command_factory(spotify_client),
    ]

    vision = Vision(camera_id=0, gesture_commands=spotify_gestures, draw=True)
    vision.activate()
