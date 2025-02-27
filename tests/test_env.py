"""
Test file for verifying that environment settings are correctly loaded.

This test ensures:
1. The `.env` file exists in the project root.
2. Essential environment variables are properly loaded into `settings`.

==== ENV VARS for Spotify ====
SPOTIFY_APPLICATION_PATH=<path_to_spotify_app>
SPOTIPY_CLIENT_ID=<your_client_id>
SPOTIPY_CLIENT_SECRET=<your_client_secret>
SPOTIPY_REDIRECT_URI=<your_redirect_uri>
SPOTIFY_USER_ID=<your_spotify_user_id>
"""

import pytest
from dotenv import find_dotenv

from src.config.settings import settings


def test_env_file_exists():
    env_path = find_dotenv()
    assert env_path, ".env file not found"


@pytest.mark.parametrize(
    "env_var",
    [
        "SPOTIFY_APPLICATION_PATH",
        "SPOTIPY_CLIENT_ID",
        "SPOTIPY_CLIENT_SECRET",
        "SPOTIPY_REDIRECT_URI",
        "SPOTIFY_USER_ID",
    ],
)
def test_spotify_env_variables_loaded(env_var):
    """
    Verify that required Spotify environment variables are loaded from the .env file.
    """
    _error_msg = f"Environment variable {env_var} not loaded. Check your `.env` file."
    assert getattr(settings, env_var) is not None, _error_msg
