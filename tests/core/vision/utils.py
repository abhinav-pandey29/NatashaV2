import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Loads an image from the given file path and converts it to RGB format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The image in RGB format.

    Raises:
        FileNotFoundError: If the image file cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Failed to load test image: {image_path}")
    image = cv2.flip(image, 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
