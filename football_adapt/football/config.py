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


def _merge_style(base: dict, over: dict, path: str = ""):
    """Deep-merge a style preset into a config dict. Every key must already exist in the config,
    so a misspelt key raises instead of silently doing nothing."""
    for k, v in over.items():
        here = f"{path}.{k}" if path else str(k)
        if k not in base:
            raise KeyError(f"style preset sets unknown config key '{here}'")
        if isinstance(v, dict) != isinstance(base[k], dict):
            raise TypeError(f"style preset key '{here}' does not match the shape of the config")
        if isinstance(v, dict):
            _merge_style(base[k], v, here)
        else:
            base[k] = v


def apply_style(cfg: Cfg, name: str) -> Cfg:
    """The config as ONE team sees it when it plays in the given style: styles.presets[name] merged in.

    A style may only change how a team plays (the `opponent` section) and its attacking / defending shape
    (`formations.phases`). It can never change the rules of the game, which both teams share."""
    presets = cfg.styles.presets.to_dict()
    if name not in presets:
        raise KeyError(f"unknown play style '{name}' (available: {', '.join(presets)})")
    preset = presets[name] or {}
    bad = [k for k in preset if k not in ("opponent", "formations")]
    bad += [f"formations.{k}" for k in preset.get("formations", {}) if k != "phases"]
    if bad:
        raise KeyError(f"play style '{name}' may only set opponent.* and formations.phases.*, not {bad}")
    d = cfg.to_dict()
    _merge_style(d, preset)
    return Cfg(d)


def load_config(path=None, overrides: dict | None = None) -> Cfg:
    """Load the YAML config and apply overrides like {"vision.radius": 15.0}."""
    with open(path or DEFAULT_PATH, "r") as f:
        d = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        _set_dotted(d, k, v)
    cfg = Cfg(d)
    for name in d.get("styles", {}).get("presets", {}):      # fail now, not mid-experiment, on a bad preset
        apply_style(cfg, name)
    return cfg
