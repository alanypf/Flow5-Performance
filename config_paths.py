"""Locate data/config files that live under Flow5-Performance/config/.

The propeller, motor, battery, plane and aero data files were moved out of the
script directory into purpose-specific subfolders:

    config/propellers/   APC PERFILES propeller tables (PER3_*.txt)
    config/motors/       motor spec XML
    config/batteries/    battery spec XML
    config/planes/       plane/airframe parameter XML
    config/aero/         Flow5 polars, T5 sweeps, VITERNA aero SDF
    config/params/       generated ArduPilot .param files

`find_config` lets the scripts keep accepting a bare filename (e.g.
"motor.xml") on the command line or as a default: a bare name is resolved
against config/<subdir>/, then by a recursive search under config/. Explicit
relative or absolute paths that already exist are returned unchanged, so
callers passing a full path (or a file outside config/) are unaffected.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "config"


def find_config(name, subdir=None):
    """Resolve a data/config file name to a usable path.

    - ``None`` is passed through unchanged.
    - absolute paths, and relative paths that already exist (relative to the
      current working directory), are returned unchanged.
    - a bare name is looked up under ``config/<subdir>/`` when *subdir* is
      given, then by a recursive search under ``config/``.
    - if nothing matches, the original name is returned so the caller raises
      its own (informative) "file not found" error.
    """
    if name is None:
        return None
    p = Path(name).expanduser()
    if p.is_absolute() or p.exists():
        return str(p)
    if subdir:
        cand = CONFIG_DIR / subdir / p.name
        if cand.exists():
            return str(cand)
    hits = sorted(CONFIG_DIR.rglob(p.name))
    if hits:
        return str(hits[0])
    return name
