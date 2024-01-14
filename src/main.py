from typing import List, Optional

import cv2

from spotify import Spotify
from vision.commands import (
    GestureCommand,
    exit_vision_command_factory,
    open_spotify_command_factory,
    play_next_track_command_factory,
    play_prev_track_command_factory,
    set_volume_command_factory,
    shuffle_saved_tracks_command_factory,
)
from vision.hands import Finger, get_hand_detector


class Vision:
    """
    Primary class for Natasha's Vision.
    """

    def __init__(
        self, gestures: List[GestureCommand] = None, draw: bool = True
    ) -> None:
        self._draw = True
        self.hand_detector = get_hand_detector(draw=draw)
        self.gesture_commands = gestures if gestures is not None else []

    def get_activated_command(self, g: List[Finger]) -> Optional[List[GestureCommand]]:
        for command in self.gesture_commands:
            if command.gesture.match(g):
                return command
        return None

    def activate(self, camera_id: int = 0, hidden: bool = False) -> None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise IOError("Cannot open webcam :(")

        while cap.isOpened():
            activated_command = None
            _, frame = cap.read()
            image = cv2.flip(frame, 1)  # Flip on horizontal axis
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            found_hands = self.hand_detector.find_hands(image)
            if found_hands:
                fingers = [
                    k for hand in found_hands for k, v in hand["fingers"].items() if v
                ]

                activated_command = self.get_activated_command(fingers)
                if activated_command:
                    activated_command(
                        hand_detector=self.hand_detector,
                        image=image,
                        found_hands=found_hands,
                        draw=self._draw,
                        cap=cap,
                        hidden=hidden,
                    )
                    if not hidden:
                        cv2.imshow(
                            "Hand Tracking", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        )
                        cv2.waitKey(1)

            if not hidden:
                cv2.imshow("Hand Tracking", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break


if __name__ == "__main__":
    spotify_client = Spotify()

    gestures = [
        exit_vision_command_factory(),
        open_spotify_command_factory(),
        shuffle_saved_tracks_command_factory(spotify_client),
        play_next_track_command_factory(spotify_client),
        play_prev_track_command_factory(spotify_client),
        set_volume_command_factory(spotify_client),
    ]

    vision = Vision(gestures=gestures, draw=True)
    vision.activate(0)
