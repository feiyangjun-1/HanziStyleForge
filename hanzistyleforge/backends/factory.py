from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BackendUnavailable, GenerationBackend
from .dir_backend import DirectoryBackend


def backend_name(cfg: dict[str, Any], override: str | None = None) -> str:
    """Resolve which backend to use, with the command line winning."""

    if override:
        return str(override).lower()
    return str(cfg.get("backend", {}).get("name", "native")).lower()


def create_backend(cfg: dict[str, Any], override: str | None = None) -> GenerationBackend:
    """Build the configured backend.

    ``native`` never reaches this function: the caller dispatches to the
    existing fusion path before constructing anything, so the built-in
    generation stack keeps working even if this module raises.
    """

    name = backend_name(cfg, override)
    backend_cfg = cfg.get("backend", {})
    if name == "dir":
        options = backend_cfg.get("dir", {})
        return DirectoryBackend(
            [Path(item) for item in options.get("candidate_dirs", [])],
            require_complete=bool(options.get("require_complete", False)),
        )
    if name == "native":
        raise BackendUnavailable(
            "The native backend is the built-in generation stack and is not constructed here."
        )
    raise BackendUnavailable(
        f"Unknown backend {name!r}. Available values: native, dir."
    )
