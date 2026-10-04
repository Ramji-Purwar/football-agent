"""5-a-side football game (no goalkeepers) with limited vision. See game.md."""
from .config import load_config, Cfg
from .env import FootballEnv

__all__ = ["load_config", "Cfg", "FootballEnv"]
