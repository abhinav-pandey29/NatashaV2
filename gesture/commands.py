"""
Gesture triggered commands
"""
import math
from abc import ABC, abstractmethod

import cv2
import numpy as np
from mediapipe.python.solutions.hands import HandLandmark

from gesture import Gesture
from gesture.hands import Finger
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
        result = self.callback(*args, **kwargs)

        return result


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

    def callback(self, *args, **kwargs):
        num_tracks = kwargs.get("num_tracks", 20)
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

    def callback(self, *args, **kwargs):
        found_hands = kwargs.get("found_hands")
        draw = kwargs.get("draw")
        image = kwargs.get("image")
        Y, X, _ = image.shape

        hands = {hand["label"]: hand for hand in found_hands}
        if "Left" in hands:
            left_wrist = hands["Left"]["landmarks"].landmark[HandLandmark.WRIST]
            originX, originY = left_wrist.x, left_wrist.y
        else:
            originX, originY = 0, right_thumb.y
        right_thumb = hands["Right"]["landmarks"].landmark[HandLandmark.THUMB_TIP]
        controlX, controlY = right_thumb.x, right_thumb.y
        origin = int(originX * X), int(originY * Y)
        control = int(controlX * X), int(controlY * Y)

        distance = math.sqrt(
            (origin[0] - control[0]) ** 2 + (origin[1] - control[1]) ** 2
        )
        distance_norm = (distance - 35) / (350 - 35) * 100
        volume = int(min(100, max(0, distance_norm)))

        # Set volume if right pinky is flicked
        right_pinky = hands["Right"]["landmarks"].landmark[HandLandmark.PINKY_TIP]
        right_ring = hands["Right"]["landmarks"].landmark[HandLandmark.RING_FINGER_DIP]
        if right_ring.x > right_pinky.x:
            print(f" - Setting volume to {volume}...")
            self.client.set_volume(volume)

        if draw:
            cv2.circle(image, control, radius=2, color=(0, 255, 255))
            cv2.line(image, control, origin, color=(0, 255, 0), thickness=2)
            cv2.putText(
                image,
                f"Volume: {volume}",
                ((origin[0] + control[0]) // 2, 20 + (origin[1] + control[1]) // 2),
                settings.CV2_FONT_TYPE,
                0.5,
                (0, 255, 0),
                1,
                settings.CV2_LINE_TYPE,
            )


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
