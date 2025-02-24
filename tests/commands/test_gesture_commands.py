from unittest.mock import MagicMock, patch

import pytest

from src.commands.gesture_commands import (
    ExitVision,
    Gesture,
    OpenSpotify,
    PlayNextTrack,
    PlayPreviousTrack,
    SetPlaybackVolume,
    ShufflePlaySavedTracks,
    exit_vision_command_factory,
    open_spotify_command_factory,
    play_next_track_command_factory,
    play_prev_track_command_factory,
    set_volume_command_factory,
    shuffle_saved_tracks_command_factory,
)
from src.config.settings import settings
from src.core.vision.hands.result import Finger
from src.integrations.spotify import Spotify


@pytest.fixture
def mock_spotify():
    """Returns a mocked Spotify client."""
    return MagicMock(spec=Spotify)


@pytest.fixture
def test_gesture():
    """Creates a test Gesture object."""
    return Gesture(Finger.RIGHT_THUMB, Finger.RIGHT_INDEX_FINGER)


@pytest.mark.parametrize(
    "factory_function, expected_type",
    [
        (exit_vision_command_factory, ExitVision),
        (open_spotify_command_factory, OpenSpotify),
        (shuffle_saved_tracks_command_factory, ShufflePlaySavedTracks),
        (play_next_track_command_factory, PlayNextTrack),
        (play_prev_track_command_factory, PlayPreviousTrack),
        (set_volume_command_factory, SetPlaybackVolume),
    ],
)
def test_factories(factory_function, expected_type, mock_spotify):
    """Ensures factory functions return the expected command instances."""
    if "spotify_client" in factory_function.__code__.co_varnames:
        instance = factory_function(mock_spotify)
    else:
        instance = factory_function()

    assert isinstance(instance, expected_type)


def test_gesture_repr(test_gesture):
    """Tests Gesture string representation."""
    assert repr(test_gesture) == "Gesture: RIGHT_THUMB,RIGHT_INDEX"


def test_gesture_match(test_gesture):
    """Tests gesture matching."""
    assert test_gesture.match([Finger.RIGHT_THUMB, Finger.RIGHT_INDEX_FINGER])
    assert not test_gesture.match([Finger.RIGHT_PINKY])
    assert not test_gesture.match([])


@patch("subprocess.Popen")
def test_open_spotify(mock_popen):
    """Tests OpenSpotify command execution."""
    cmd = open_spotify_command_factory()
    cmd()
    mock_popen.assert_called_once_with(settings.SPOTIFY_APPLICATION_PATH)


def test_shuffle_play(mock_spotify):
    """Tests ShufflePlaySavedTracks command execution."""
    mock_spotify.get_saved_tracks.return_value = ["track1", "track2"]

    cmd = shuffle_saved_tracks_command_factory(mock_spotify)
    cmd(num_tracks=2)

    mock_spotify.get_saved_tracks.assert_called_once_with(2)
    mock_spotify.shuffle_play.assert_called_once_with("track1", "track2")


def test_play_next_track(mock_spotify):
    """Tests PlayNextTrack command execution."""
    cmd = play_next_track_command_factory(mock_spotify)
    cmd()
    mock_spotify.next_track.assert_called_once()


def test_play_previous_track(mock_spotify):
    """Tests PlayPreviousTrack command execution."""
    cmd = play_prev_track_command_factory(mock_spotify)
    cmd()
    mock_spotify.previous_track.assert_called_once()
