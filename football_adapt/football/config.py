"""Config loading: YAML file plus dotted-key overrides, with attribute access."""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Cfg:
    """Wrapper so nested config can be used as cfg.pitch.length."""

    def __init__(self, d: dict):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name):
        try:
            v = self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __getitem__(self, name):
        v = self._d[name]
        return Cfg(v) if isinstance(v, dict) else v

    def get(self, name, default=None):
        return self[name] if name in self._d else default

    def to_dict(self) -> dict:
        return copy.deepcopy(self._d)


def _set_dotted(d: dict, key: str, value):
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def load_config(path=None, overrides: dict | None = None) -> Cfg:
    """Load the YAML config and apply overrides like {"vision.radius": 15.0}."""
    with open(path or DEFAULT_PATH, "r") as f:
        d = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        _set_dotted(d, k, v)
    return Cfg(d)
