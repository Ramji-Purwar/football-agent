"""5-a-side football game (no goalkeepers) with limited vision. See game.md."""
from .config import load_config, apply_style, Cfg
from .env import FootballEnv

__all__ = ["load_config", "apply_style", "Cfg", "FootballEnv"]
