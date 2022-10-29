class Settings:
    """
    Configuration settings for Natasha.
    """

    SPOTIPY_CLIENT_ID = "c10601a9449f401da5b8490fbfa74d3b"
    SPOTIPY_CLIENT_SECRET = "fab5db69e6cb4db5b6fff6e59d81a45a"
    SPOTIPY_REDIRECT_URI = "http://127.0.0.1:9090"
    SPOTIFY_USER_ID = "31e3jjllwbuuqwqb222m33w5svyq"
    SPOTIFY_SCOPE = ",".join(
        [
            "user-read-private",
            "user-follow-modify",
            "user-follow-read",
            "user-library-modify",
            "user-library-read",
            "playlist-modify-private",
            "playlist-read-private",
            "user-top-read",
            "user-read-currently-playing",
            "user-read-recently-played",
            "user-modify-playback-state",
            "user-read-playback-state",
        ]
    )

settings = Settings()
