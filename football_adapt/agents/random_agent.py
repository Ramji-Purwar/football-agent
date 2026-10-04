"""Random baseline: picks uniformly among VALID actions (uses the action mask)."""
import numpy as np


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> dict:
        mask = obs["mask"]
        return {i: int(self.rng.choice(np.flatnonzero(mask[i]))) for i in range(mask.shape[0])}
