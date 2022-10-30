"""
Gesture triggered commands
"""
from abc import ABC, abstractmethod

import cv2
import numpy as np

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
        fingers=[
            Finger.RIGHT_PINKY,
            Finger.RIGHT_THUMB,
            Finger.LEFT_INDEX_FINGER,
            Finger.LEFT_MIDDLE_FINGER,
        ]
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
        fingers=[
            Finger.RIGHT_MIDDLE_FINGER,
            Finger.RIGHT_THUMB,
        ]
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
        fingers=[
            Finger.RIGHT_PINKY,
            Finger.RIGHT_THUMB,
            Finger.LEFT_THUMB,
        ]
    )

    def __init__(self, spotify_client: Spotify):
        self.client = spotify_client

    def callback(self, *args, **kwargs):
        return self.client.previous_track()


def shuffle_saved_tracks_command_factory(spotify_client: Spotify):
    """Factory for ShufflePlaySavedTracks command."""
    return ShufflePlaySavedTracks(spotify_client=spotify_client)


def play_next_track_command_factory(spotify_client: Spotify):
    """Factory for PlayNextTrack command."""
    return PlayNextTrack(spotify_client=spotify_client)


def play_prev_track_command_factory(spotify_client: Spotify):
    """Factory for PlayPreviousTrack command."""
    return PlayPreviousTrack(spotify_client=spotify_client)
