"""
Gesture triggered commands
"""
from abc import ABC, abstractmethod

import cv2
import numpy as np

from gesture import Gesture
from settings import settings
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
