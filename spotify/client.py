"""
Spotify service and actions.
"""
import random
from typing import List, Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from settings import settings
from spotify.entity import Artist, AudioFeatures, PlaybackDevice, TrackItem
from utils import get_chunks


class Spotify:
    """Wrapper for Spotify service."""

    client: spotipy.Spotify = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=settings.SPOTIPY_CLIENT_ID,
            client_secret=settings.SPOTIPY_CLIENT_SECRET,
            redirect_uri=settings.SPOTIPY_REDIRECT_URI,
            scope=settings.SPOTIFY_SCOPE,
        )
    )

    def primary_device_id(self) -> Optional[str]:
        devices = self.get_devices()
        if devices:
            return devices.pop(0).id
        else:
            return None

    def get_devices(self) -> List[PlaybackDevice]:
        """
        Get users devices.

        Active device (if any) is placed first. In case, none of the devices
        are active then "Computer" devices are prioritized over "Smartphone".
        """
        devices = self.client.devices()
        devices = [PlaybackDevice.parse_data(device) for device in devices["devices"]]
        devices.sort(key=lambda d: (not d.is_active, d.type))
        return devices

    def get_active_device(self) -> Optional[PlaybackDevice]:
        for device in self.get_devices():
            if device.is_active:
                return device
        return None

    def get_followed_artists(self, limit: int) -> List[Artist]:
        """
        Get list of followed artists.
        Iterates over paginated responses automagically, if required.
        """
        followed_artists = []
        step_size = min(50, limit)
        after = None
        while len(followed_artists) < limit:
            artists = self.client.current_user_followed_artists(
                limit=step_size, after=after
            )
            followed_artists.extend(artists["artists"]["items"])

            if len(followed_artists) < artists["artists"]["total"]:
                after = followed_artists[-1]["id"]
            else:
                break

        return [Artist.parse_data(artist) for artist in followed_artists]

    def get_saved_tracks(self, limit: int) -> List[TrackItem]:
        """
        Get list of saved tracks.
        Iterates over paginated responses automagically, if required.
        """
        saved_tracks = []
        step_size = min(50, limit)
        offset = 0
        while len(saved_tracks) < limit:
            tracks = self.client.current_user_saved_tracks(
                limit=step_size, offset=offset
            )
            saved_tracks.extend(tracks["items"])

            if len(saved_tracks) < tracks["total"]:
                offset += step_size
            else:
                break

        return [TrackItem.parse_data(track["track"]) for track in saved_tracks]

    # TODO: Investigate if at all possible to get more than 50
    # recently played tracks.
    def get_recently_played(self, limit: int = 50) -> List[TrackItem]:
        if limit > 50:
            raise ValueError("Limit MUST be less than 50.")
        # recently_played = []
        # step_size = min(50, limit)
        # before = None
        # while len(recently_played) < limit:
        #     tracks = self.client.current_user_recently_played(
        #         limit=step_size, before=before
        #     )
        #     recently_played.extend(tracks["items"])
        #     if len(recently_played) < limit and tracks["next"]:
        #         before = tracks["cursors"]["before"]
        #     else:
        #         break
        recently_played = self.client.current_user_recently_played(limit=int(limit))
        return [
            TrackItem.parse_data({**track["track"], "played_at": track["played_at"]})
            for track in recently_played["items"]
        ]

    def get_audio_features(self, tracks: List[TrackItem]) -> List[AudioFeatures]:
        """
        Get audio features for the given tracks.
        Handles size restrictions using batch requests, if required.
        """
        track_ids = [track.id for track in tracks]
        audio_features = []
        for chunk in get_chunks(track_ids, 100):  # 100 = Max allowed tracks
            chunk_features = self.client.audio_features(chunk)
            audio_features.extend(chunk_features)

        return [AudioFeatures.parse_data(features) for features in audio_features]

    def set_volume(self, volume_percent: int) -> None:
        """Sets playback volume on active device."""
        if volume_percent < 0 or volume_percent > 100:
            raise ValueError("Volume must be between 0 and 100!")

        active_device = self.get_active_device()
        if active_device:
            self.client.volume(int(volume_percent), active_device.id)

    def add_to_queue(self, *tracks: TrackItem) -> None:
        for track in tracks:
            print(f"📑 Queueing -> {track.name}")
            self.client.add_to_queue(track.uri)

    def play(self, *tracks: TrackItem) -> None:
        return self.client.start_playback(
            device_id=self.primary_device_id(),
            uris=[track.uri for track in tracks],
        )

    def next_track(self) -> None:
        return self.client.next_track(device_id=self.primary_device_id())

    def previous_track(self) -> None:
        return self.client.previous_track(device_id=self.primary_device_id())

    def shuffle_play(self, *tracks: TrackItem) -> None:
        tracks = list(tracks)
        random.shuffle(tracks)
        return self.play(*tracks)
