"""
Utilities for Natasha
"""
from typing import Any, List


def chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
    """Break a list into chunks of size N"""
    return [lst[i : i + n] for i in range(0, len(lst), n)]
