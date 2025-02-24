"""
Tests for Spotify service and actions.
"""

from unittest.mock import Mock

import pytest

from src.integrations.spotify import Spotify
from src.integrations.spotify.entity import PlaybackDevice

MOCK_DEVICE_RESPONSE_ACTIVE_PHONE = {
    "id": "device-2",
    "is_active": True,
    "name": "Phone",
    "type": "Smartphone",
    "volume_percent": 30,
}
MOCK_DEVICE_RESPONSE_INACTIVE_LAPTOP = {
    "id": "device-1",
    "is_active": False,
    "name": "Laptop",
    "type": "Computer",
    "volume_percent": 50,
}
MOCK_DEVICE_RESPONSE_INACTIVE_SPEAKER = {
    "id": "device-3",
    "is_active": False,
    "name": "Speaker",
    "type": "Speaker",
    "volume_percent": 75,
}
MOCK_DEVICE_LIST_MIXED = {
    "devices": [
        MOCK_DEVICE_RESPONSE_INACTIVE_LAPTOP,
        MOCK_DEVICE_RESPONSE_ACTIVE_PHONE,
        MOCK_DEVICE_RESPONSE_INACTIVE_SPEAKER,
    ]
}
MOCK_DEVICE_LIST_NO_ACTIVE = {
    "devices": [
        MOCK_DEVICE_RESPONSE_INACTIVE_SPEAKER,
        MOCK_DEVICE_RESPONSE_INACTIVE_LAPTOP,
    ]
}
MOCK_DEVICE_LIST_EMPTY = {"devices": []}


@pytest.fixture
def spotify_instance():
    """Fixture for a Spotify instance with a mocked spotipy.client."""
    spotify = Spotify()
    spotify.client = Mock()
    return spotify


def test_get_devices(spotify_instance):
    """Assert get_devices correctly parses and sorts devices."""
    spotify_instance.client.devices.return_value = MOCK_DEVICE_LIST_MIXED
    devices = spotify_instance.get_devices()

    # Ensure devices are sorted (active first, then Computer > Smartphone > Others)
    assert len(devices) == 3
    assert isinstance(devices[0], PlaybackDevice)
    assert devices[0].id == "device-2"  # Active device should be first
    assert devices[1].id == "device-1"  # Computer should come before Smartphone
    assert devices[2].id == "device-3"  # Speaker at the end


def test_primary_device_id(spotify_instance):
    """Assert primary_device_id returns the first device's ID or None if no devices exist."""
    # Case: Devices exist, active device available
    spotify_instance.client.devices.return_value = MOCK_DEVICE_LIST_MIXED
    assert spotify_instance.primary_device_id() == "device-2"  # First active device

    # Case: No active device
    spotify_instance.client.devices.return_value = MOCK_DEVICE_LIST_NO_ACTIVE
    assert spotify_instance.primary_device_id() == "device-1"  # Computer prioritized

    # Case: No devices at all
    spotify_instance.client.devices.return_value = MOCK_DEVICE_LIST_EMPTY
    assert spotify_instance.primary_device_id() is None


def test_get_active_device(spotify_instance):
    """Assert get_active_device returns the active device or None if no devices are active."""
    # Case: One active device
    spotify_instance.client.devices.return_value = MOCK_DEVICE_LIST_MIXED
    active_device = spotify_instance.get_active_device()
    assert active_device is not None
    assert active_device.id == "device-2"

    # Case: No active devices
    spotify_instance.client.devices.return_value = MOCK_DEVICE_LIST_NO_ACTIVE
    assert spotify_instance.get_active_device() is None
