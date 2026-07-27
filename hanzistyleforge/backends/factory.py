from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BackendUnavailable, GenerationBackend
from .dir_backend import DirectoryBackend
from .zi2zi_jit import Zi2ziJitBackend


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
    if name == "zi2zi-jit":
        options = backend_cfg.get("zi2zi_jit", {})
        font_label = options.get("font_label", None)
        return Zi2ziJitBackend(
            options.get("repo_dir", ""),
            options.get("checkpoint", ""),
            python_executable=options.get("python_executable", "") or None,
            model=options.get("model", "") or None,
            sampling_method=str(options.get("sampling_method", "ab2")),
            num_sampling_steps=int(options.get("num_sampling_steps", 20)),
            cfg_scale=float(options.get("cfg_scale", 2.6)),
            batch_size=int(options.get("batch_size", 16)),
            font_label=None if font_label is None else int(font_label),
            chunk_size=int(options.get("chunk_size", 2048)),
            style_pool_size=int(options.get("style_pool_size", 64)),
            timeout_seconds=int(options.get("timeout_seconds", 0)),
            disable_torch_compile=bool(options.get("disable_torch_compile", True)),
        )
    if name == "native":
        raise BackendUnavailable(
            "The native backend is the built-in generation stack and is not constructed here."
        )
    raise BackendUnavailable(
        f"Unknown backend {name!r}. Available values: native, dir, zi2zi-jit."
    )
