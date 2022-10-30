from typing import List

from gesture.hands import Finger


class Gesture:
    """Hand Gesture object."""

    def __init__(self, *fingers: Finger):
        self.fingers = list(fingers)

    def __repr__(self):
        return "Gesture: " + ",".join(self.fingers)

    def match(self, g: List[Finger]):
        return set(self.fingers) == set(g)
