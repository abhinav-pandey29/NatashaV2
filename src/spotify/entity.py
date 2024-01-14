"""
Spotify entities and related enums.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlmodel import SQLModel


class _BaseSpotifyEntity(SQLModel):
    _translation_paths: Dict[str, Tuple[str]] = {}

    @staticmethod
    def parse_translations(data_dict: dict, path: Tuple[str]) -> Optional[dict]:
        for key in path:
            if key not in data_dict:
                print(key, "not in data dict:\n", data_dict)
                return None
            data_dict = data_dict[key]
        return data_dict

    @classmethod
    def parse_data(cls, data: dict) -> "_BaseSpotifyEntity":
        required_keys = {field: data.get(field) for field in cls.__fields__}
        nested_keys = {
            key: cls.parse_translations(data, path)
            for key, path in cls._translation_paths.items()
        }
        required_keys.update(nested_keys)
        return cls.parse_obj(required_keys)


class TrackItem(_BaseSpotifyEntity):
    id: str
    uri: str
    name: str
    popularity: int
    duration_ms: str
    preview_url: str = None
    spotify_url: str
    played_at: Optional[datetime] = None

    _translation_paths = {"spotify_url": ("external_urls", "spotify")}


class Artist(_BaseSpotifyEntity):
    id: str
    uri: str
    name: str
    popularity: int
    followers: int
    genres: List[str] = []
    spotify_url: str

    _translation_paths = {
        "spotify_url": ("external_urls", "spotify"),
        "followers": ("followers", "total"),
    }


class SpotifyPlaylist(_BaseSpotifyEntity):
    id: str = ""
    uri: str = ""
    name: str
    public: bool = False
    collaborative: bool = False
    description: str = ""
    spotify_url: str = ""

    _translation_paths = {"spotify_url": ("external_urls", "spotify")}


class PlaybackDevice(_BaseSpotifyEntity):
    id: str
    name: str
    type: str
    is_active: bool
    volume: float

    _translation_paths = {"volume": ("volume_percent",)}


class AudioFeatures(_BaseSpotifyEntity):
    track_id: str = None
    danceability: str
    energy: str
    loudness: str
    key: str
    mode: str
    speechiness: str
    acousticness: str
    instrumentalness: str
    liveness: str
    valence: str
    tempo: str
    duration_ms: str
    time_signature: str

    _translation_paths = {"track_id": ("id",)}
