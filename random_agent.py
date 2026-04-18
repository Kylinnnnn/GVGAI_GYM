from __future__ import annotations

from random import randint
from typing import Any


class Agent:
    """Random baseline with the same interface as the original Agent.py."""

    def __init__(self) -> None:
        self.name = "randomAgent"

    def act(self, stateObs: Any, actions: Any) -> int:
        del stateObs
        try:
            action_count = len(actions)
        except Exception:
            action_count = 1
        if action_count <= 0:
            return 0
        return int(randint(0, action_count - 1))


# Compatibility alias for newer-style imports.
RandomAgent = Agent
