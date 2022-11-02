"""
Gesture triggered commands
"""
import math
import subprocess
from abc import ABC, abstractmethod

import cv2
import numpy as np
from mediapipe.python.solutions.hands import HandLandmark

from gesture import Gesture
from gesture.hands import Finger, HandDetector
from settings import settings
from spotify import Spotify


class GestureCommand(ABC):
    """Base class for gesture triggered commands."""

    name: str = "Command"
    gesture: Gesture

    @abstractmethod
    def callback(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        print(f"{self.name} triggered by {self.gesture}")
        if kwargs.get("draw") == True and kwargs.get("image") is not None:
            image = kwargs.get("image")
            assert isinstance(image, np.ndarray)

            # display notification in center
            text = f"{self.name} activated"
            textsize = cv2.getTextSize(text, settings.CV2_FONT_TYPE, 0.5, 1)[0]
            textX = (image.shape[1] - textsize[0]) // 2
            cv2.putText(
                image,
                text=text,
                org=(textX, 20),
                fontFace=settings.CV2_FONT_TYPE,
                fontScale=0.5,
                color=settings.CV2_TEXT_COLOR,
                thickness=1,
                lineType=settings.CV2_LINE_TYPE,
            )
        try:
            result = self.callback(*args, **kwargs)
        except Exception as e:
            print(f" - Error occured when executing Command {self.name}:\n {e}")
            result = None

        return result


class OpenSpotify(GestureCommand):
    """
    Command: Open spotify desktop application.
    """

    name = "OpenSpotify"
    gesture = Gesture(
        Finger.RIGHT_PINKY,
        Finger.RIGHT_THUMB,
        Finger.RIGHT_INDEX_FINGER,
        Finger.LEFT_PINKY,
        Finger.LEFT_THUMB,
        Finger.LEFT_INDEX_FINGER,
    )

    def callback(self, *args, **kwargs) -> None:
        return subprocess.Popen(settings.SPOTIFY_APPLICATION_PATH)


class ExitVision(GestureCommand):
    """
    Command: Exit Vision.
    """

    name = "ExitVision"
    gesture = Gesture(
        Finger.RIGHT_THUMB,
        Finger.RIGHT_PINKY,
        Finger.LEFT_THUMB,
        Finger.LEFT_MIDDLE_FINGER,
    )

    def callback(self, cap: cv2.VideoCapture, *args, **kwargs) -> None:
        assert isinstance(cap, cv2.VideoCapture)
        cap.release()


class ShufflePlaySavedTracks(GestureCommand):
    """
    Command: Shuffle play most recently saved tracks.

    Parameters:
      - num_tracks: int - Number of saved tracks to play
    """

    name = "ShufflePlaySavedTracks"
    gesture = Gesture(
        Finger.RIGHT_PINKY,
        Finger.RIGHT_THUMB,
        Finger.LEFT_INDEX_FINGER,
        Finger.LEFT_MIDDLE_FINGER,
    )

    def __init__(self, spotify_client: Spotify):
        self.client = spotify_client

    def callback(self, num_tracks: int = 20, *args, **kwargs):
        saved_tracks = self.client.get_saved_tracks(num_tracks)
        self.client.shuffle_play(*saved_tracks)


class PlayNextTrack(GestureCommand):
    """
    Command: Skips to next track in the user's queue.
    """

    name = "PlayNextTrack"
    gesture = Gesture(
        Finger.RIGHT_MIDDLE_FINGER,
        Finger.RIGHT_THUMB,
    )

    def __init__(self, spotify_client: Spotify):
        self.client = spotify_client

    def callback(self, *args, **kwargs):
        return self.client.next_track()


class PlayPreviousTrack(GestureCommand):
    """
    Command: Skips to previous track in the user's queue.
    """

    name = "PlayPreviousTrack"
    gesture = Gesture(
        Finger.RIGHT_PINKY,
        Finger.RIGHT_THUMB,
        Finger.LEFT_THUMB,
    )

    def __init__(self, spotify_client: Spotify):
        self.client = spotify_client

    def callback(self, *args, **kwargs):
        return self.client.previous_track()


class SetPlaybackVolume(GestureCommand):
    """
    Command: Set Playback Volume
    """

    name = "SetPlaybackVolume"
    gesture = Gesture(
        Finger.RIGHT_THUMB,
        Finger.RIGHT_MIDDLE_FINGER,
        Finger.RIGHT_RING_FINGER,
        Finger.RIGHT_PINKY,
    )

    def __init__(self, spotify_client: Spotify):
        self.client = spotify_client

    def callback(
        self,
        hand_detector: HandDetector,
        cap: cv2.VideoCapture,
        draw: bool = False,
        *args,
        **kwargs,
    ):
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.flip(frame, 1)  # Flip on horizontal axis
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            found_hands = hand_detector.find_hands(image)
            if found_hands:
                fingers = [
                    k for hand in found_hands for k, v in hand["fingers"].items() if v
                ]
                if not self.gesture.match(fingers):
                    print(f"  - {self.name} exited...")
                    break

                Y, X, _ = image.shape
                hands = {hand["label"]: hand for hand in found_hands}
                right_thumb = hands["Right"]["landmarks"].landmark[
                    HandLandmark.THUMB_TIP
                ]
                controlX, controlY = right_thumb.x, right_thumb.y
                if "Left" in hands:
                    left_wrist = hands["Left"]["landmarks"].landmark[HandLandmark.WRIST]
                    originX, originY = left_wrist.x, left_wrist.y
                else:
                    originX, originY = 0, right_thumb.y
                origin = int(originX * X), int(originY * Y)
                control = int(controlX * X), int(controlY * Y)

                distance = math.sqrt(
                    (origin[0] - control[0]) ** 2 + (origin[1] - control[1]) ** 2
                )
                distance_norm = (distance - 35) / (350 - 35) * 100
                volume = int(min(100, max(0, distance_norm)))

                # Set volume if right pinky is flicked
                right_pinky = hands["Right"]["landmarks"].landmark[
                    HandLandmark.PINKY_TIP
                ]
                right_ring = hands["Right"]["landmarks"].landmark[
                    HandLandmark.RING_FINGER_DIP
                ]
                if right_ring.x > right_pinky.x:
                    print(f" - Setting volume to {volume}...")
                    # Set volume and exit from command loop
                    return self.client.set_volume(volume)

                if draw:
                    cv2.circle(image, control, radius=2, color=(0, 255, 255))
                    cv2.line(image, control, origin, color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        image,
                        f"Volume: {volume}",
                        (
                            (origin[0] + control[0]) // 2,
                            20 + (origin[1] + control[1]) // 2,
                        ),
                        settings.CV2_FONT_TYPE,
                        0.5,
                        (0, 255, 0),
                        1,
                        settings.CV2_LINE_TYPE,
                    )
                    cv2.imshow("Hand Tracking", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break


def exit_vision_command_factory():
    """Factory for ExitVision command."""
    return ExitVision()


def open_spotify_command_factory():
    """Factory for OpenSpotify command."""
    return OpenSpotify()


def shuffle_saved_tracks_command_factory(spotify_client: Spotify):
    """Factory for ShufflePlaySavedTracks command."""
    return ShufflePlaySavedTracks(spotify_client=spotify_client)


def play_next_track_command_factory(spotify_client: Spotify):
    """Factory for PlayNextTrack command."""
    return PlayNextTrack(spotify_client=spotify_client)


def play_prev_track_command_factory(spotify_client: Spotify):
    """Factory for PlayPreviousTrack command."""
    return PlayPreviousTrack(spotify_client=spotify_client)


def set_volume_command_factory(spotify_client: Spotify):
    """Factory for SetPlaybackVolume command."""
    return SetPlaybackVolume(spotify_client=spotify_client)
