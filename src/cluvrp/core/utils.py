"""Small utility helpers."""

from __future__ import annotations

import random
from typing import Dict, List


def weighted_choice(items: List[str], weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(weights[item] for item in items)
    x = rng.random() * total
    acc = 0.0
    for item in items:
        acc += weights[item]
        if x <= acc:
            return item
    return items[-1]