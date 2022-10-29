"""
Utilities for Natasha
"""
from typing import Any, List


def get_chunks(l: List[Any], n: int) -> List[List[Any]]:
    """Break a list into chunks of size N"""
    return [l[i : i + n] for i in range(0, len(l), n)]
