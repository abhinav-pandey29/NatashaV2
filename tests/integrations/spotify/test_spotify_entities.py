"""
Tests for Spotify entities.
"""

import src.integrations.spotify.entity as entity

# Mock response stubs from Spotify API
TRACK_STUB = {
    "album": {
        "album_type": "compilation",
        "total_tracks": 9,
        "available_markets": ["CA", "BR", "IT"],
        "external_urls": {"spotify": "string"},
        "href": "string",
        "id": "2up3OPMp9Tb4dAKM2erWXQ",
        "images": [
            {
                "url": "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228",
                "height": 300,
                "width": 300,
            }
        ],
        "name": "string",
        "release_date": "1981-12",
        "release_date_precision": "year",
        "restrictions": {"reason": "market"},
        "type": "album",
        "uri": "spotify:album:2up3OPMp9Tb4dAKM2erWXQ",
        "artists": [
            {
                "external_urls": {"spotify": "string"},
                "href": "string",
                "id": "string",
                "name": "string",
                "type": "artist",
                "uri": "string",
            }
        ],
    },
    "artists": [
        {
            "external_urls": {"spotify": "string"},
            "href": "string",
            "id": "string",
            "name": "string",
            "type": "artist",
            "uri": "string",
        }
    ],
    "available_markets": ["string"],
    "disc_number": 0,
    "duration_ms": 11552,
    "explicit": False,
    "external_ids": {"isrc": "string", "ean": "string", "upc": "string"},
    "external_urls": {"spotify": "test-track-spotify-url"},
    "href": "string",
    "id": "test-track-id",
    "is_playable": False,
    "linked_from": {},
    "restrictions": {"reason": "string"},
    "name": "test-track-name",
    "popularity": 327,
    "preview_url": "test-track-preview-url",
    "track_number": 0,
    "type": "track",
    "uri": "test-track-uri",
    "is_local": False,
}
ARTIST_STUB = {
    "external_urls": {"spotify": "test-artist-spotify-url"},
    "followers": {"href": "string", "total": 11552},
    "genres": ["Prog rock", "Grunge"],
    "href": "string",
    "id": "test-artist-id",
    "images": [
        {
            "url": "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228",
            "height": 300,
            "width": 300,
        }
    ],
    "name": "test-artist-name",
    "popularity": 327,
    "type": "artist",
    "uri": "test-artist-uri",
}
PLAYLIST_STUB = {
    "collaborative": True,
    "description": "test-playlist-description",
    "external_urls": {"spotify": "test-playlist-spotify-url"},
    "followers": {"href": "string", "total": 0},
    "href": "string",
    "id": "test-playlist-id",
    "images": [
        {
            "url": "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228",
            "height": 300,
            "width": 300,
        }
    ],
    "name": "test-playlist-name",
    "owner": {
        "external_urls": {"spotify": "string"},
        "followers": {"href": "string", "total": 0},
        "href": "string",
        "id": "string",
        "type": "user",
        "uri": "string",
        "display_name": "string",
    },
    "public": True,
    "snapshot_id": "string",
    "tracks": {
        "href": "https://api.spotify.com/v1/me/shows?offset=0&limit=20",
        "limit": 20,
        "next": "https://api.spotify.com/v1/me/shows?offset=1&limit=1",
        "offset": 0,
        "previous": "https://api.spotify.com/v1/me/shows?offset=1&limit=1",
        "total": 4,
        "items": [
            {
                "added_at": "string",
                "added_by": {
                    "external_urls": {"spotify": "string"},
                    "followers": {"href": "string", "total": 0},
                    "href": "string",
                    "id": "string",
                    "type": "user",
                    "uri": "string",
                },
                "is_local": False,
                "track": {
                    "album": {
                        "album_type": "compilation",
                        "total_tracks": 9,
                        "available_markets": ["CA", "BR", "IT"],
                        "external_urls": {"spotify": "string"},
                        "href": "string",
                        "id": "2up3OPMp9Tb4dAKM2erWXQ",
                        "images": [
                            {
                                "url": "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228",
                                "height": 300,
                                "width": 300,
                            }
                        ],
                        "name": "string",
                        "release_date": "1981-12",
                        "release_date_precision": "year",
                        "restrictions": {"reason": "market"},
                        "type": "album",
                        "uri": "spotify:album:2up3OPMp9Tb4dAKM2erWXQ",
                        "artists": [
                            {
                                "external_urls": {"spotify": "string"},
                                "href": "string",
                                "id": "string",
                                "name": "string",
                                "type": "artist",
                                "uri": "string",
                            }
                        ],
                    },
                    "artists": [
                        {
                            "external_urls": {"spotify": "string"},
                            "href": "string",
                            "id": "string",
                            "name": "string",
                            "type": "artist",
                            "uri": "string",
                        }
                    ],
                    "available_markets": ["string"],
                    "disc_number": 0,
                    "duration_ms": 0,
                    "explicit": False,
                    "external_ids": {
                        "isrc": "string",
                        "ean": "string",
                        "upc": "string",
                    },
                    "external_urls": {"spotify": "string"},
                    "href": "string",
                    "id": "string",
                    "is_playable": False,
                    "linked_from": {},
                    "restrictions": {"reason": "string"},
                    "name": "string",
                    "popularity": 0,
                    "preview_url": "string",
                    "track_number": 0,
                    "type": "track",
                    "uri": "string",
                    "is_local": False,
                },
            }
        ],
    },
    "type": "string",
    "uri": "test-playlist-uri",
}
DEVICE_STUB = {
    "id": "test-device-id",
    "is_active": True,
    "is_private_session": False,
    "is_restricted": False,
    "name": "test-device-name",
    "type": "test-device-type",
    "volume_percent": 79,
    "supports_volume": False,
}


def test_spotify_track_entity():
    """Verify TrackItem entity correctly parses a Spotify track response."""
    track_entity = entity.TrackItem.parse_data(TRACK_STUB)

    assert isinstance(track_entity, entity.TrackItem)

    assert track_entity.id == "test-track-id"
    assert track_entity.uri == "test-track-uri"
    assert track_entity.name == "test-track-name"
    assert track_entity.popularity == 327
    assert track_entity.duration_ms == 11552
    assert track_entity.preview_url == "test-track-preview-url"
    assert track_entity.spotify_url == "test-track-spotify-url"
    assert track_entity.played_at is None


def test_spotify_artist_entity():
    """Verify Artist entity correctly parses a Spotify artist response."""
    artist_entity = entity.Artist.parse_data(ARTIST_STUB)

    assert isinstance(artist_entity, entity.Artist)

    assert artist_entity.id == "test-artist-id"
    assert artist_entity.uri == "test-artist-uri"
    assert artist_entity.name == "test-artist-name"
    assert artist_entity.popularity == 327
    assert artist_entity.followers == 11552
    assert artist_entity.spotify_url == "test-artist-spotify-url"
    assert artist_entity.genres == ["Prog rock", "Grunge"]


def test_spotify_playlist_entity():
    """Verify SpotifyPlaylist entity correctly parses a Spotify playlist response."""
    playlist_entity = entity.SpotifyPlaylist.parse_data(PLAYLIST_STUB)

    assert isinstance(playlist_entity, entity.SpotifyPlaylist)

    assert playlist_entity.id == "test-playlist-id"
    assert playlist_entity.uri == "test-playlist-uri"
    assert playlist_entity.name == "test-playlist-name"
    assert playlist_entity.description == "test-playlist-description"
    assert playlist_entity.spotify_url == "test-playlist-spotify-url"
    assert playlist_entity.public is True
    assert playlist_entity.collaborative is True


def test_spotify_device_entity():
    """Verify PlaybackDevice entity correctly parses a Spotify device response."""
    device_entity = entity.PlaybackDevice.parse_data(DEVICE_STUB)

    assert isinstance(device_entity, entity.PlaybackDevice)

    assert device_entity.id == "test-device-id"
    assert device_entity.name == "test-device-name"
    assert device_entity.type == "test-device-type"
    assert device_entity.is_active is True
    assert device_entity.volume == 79
