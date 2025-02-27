import os

import cv2
from dotenv import find_dotenv, load_dotenv
from mediapipe.python.solutions import drawing_utils

load_dotenv(find_dotenv())


class Settings:
    """
    Configuration settings for Natasha.
    """

    SPOTIFY_APPLICATION_PATH = os.getenv("SPOTIFY_APPLICATION_PATH")
    SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
    SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
    SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")
    SPOTIFY_USER_ID = os.getenv("SPOTIFY_USER_ID")
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

    CV2_FONT_TYPE = cv2.FONT_HERSHEY_DUPLEX
    CV2_TEXT_COLOR = (255, 255, 255)  # white
    CV2_LINE_TYPE = cv2.LINE_AA

    HAND_JOINT_SPEC = drawing_utils.DrawingSpec(
        color=(255, 69, 0), thickness=2, circle_radius=2
    )
    HAND_CONNECTION_SPEC = drawing_utils.DrawingSpec(
        color=(255, 69, 0), thickness=2, circle_radius=2
    )
    BBOX_PADDING = 15


settings = Settings()
