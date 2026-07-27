from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from .base import (
    BackendRequest,
    BackendResult,
    BackendUnavailable,
    read_candidate_dir,
)


class DirectoryBackend:
    """Read candidates from directories that already contain generated images.

    This backend runs no model.  It exists so the whole post-processing chain
    (candidate selection, IDS component checks, QA, refinement, vectorization,
    TTF build) can be exercised end to end without depending on any third-party
    generator being installed and working, and so images produced by a manual
    run of some other tool can be fed into the pipeline.

    Each configured directory is one candidate: three directories give the
    selection stage three candidates per glyph.
    """

    name = "dir"

    def __init__(
        self,
        candidate_dirs: Sequence[str | Path],
        *,
        require_complete: bool = False,
    ) -> None:
        self._candidate_dirs = [Path(item) for item in candidate_dirs]
        self._require_complete = bool(require_complete)

    def preflight(self) -> None:
        if not self._candidate_dirs:
            raise BackendUnavailable(
                "The dir backend has no candidate directories configured. "
                "Set backend.dir.candidate_dirs to one or more directories of "
                "U+XXXX.png images, one directory per candidate."
            )
        missing = [path for path in self._candidate_dirs if not path.is_dir()]
        if missing:
            listed = "\n  ".join(str(path) for path in missing)
            raise BackendUnavailable(
                "The dir backend cannot find these candidate directories:\n  "
                + listed
                + "\nCreate them, or correct backend.dir.candidate_dirs."
            )
        empty: list[Path] = []
        for path in self._candidate_dirs:
            try:
                if not read_candidate_dir(path):
                    empty.append(path)
            except ValueError as exc:
                raise BackendUnavailable(str(exc)) from exc
        if empty:
            listed = "\n  ".join(str(path) for path in empty)
            raise BackendUnavailable(
                "These candidate directories contain no file whose name carries a "
                "codepoint:\n  "
                + listed
                + "\nImages must be named U+XXXX.png (zi2zi-JiT's 0000_U+XXXX.png is also accepted)."
            )

    def generate(self, request: BackendRequest) -> BackendResult:
        self.preflight()
        requested = set(request.codepoints())
        per_directory: list[dict[str, Any]] = []
        missing_everywhere = set(requested)

        for path in self._candidate_dirs:
            available = read_candidate_dir(path)
            covered = requested & set(available)
            missing_everywhere -= covered
            per_directory.append(
                {
                    "directory": str(path.resolve()),
                    "image_count": len(available),
                    "requested_covered": len(covered),
                    "extra_not_requested": len(set(available) - requested),
                }
            )

        if missing_everywhere and self._require_complete:
            sample = ", ".join(f"U+{cp:04X}" for cp in sorted(missing_everywhere)[:12])
            raise BackendUnavailable(
                f"{len(missing_everywhere)} requested glyphs are absent from every candidate "
                f"directory (for example {sample}). Provide them, or set "
                "backend.dir.require_complete=false to let the reference fallback cover them."
            )

        metadata: dict[str, Any] = {
            "backend": self.name,
            "requested_glyphs": len(requested),
            "candidate_dirs": per_directory,
            "missing_from_all_candidates": len(missing_everywhere),
            # The caller resolves these through the existing fail-complete
            # reference fallback, exactly as it already does for a glyph whose
            # learned candidates all fail the topology gate.
            "missing_sample": [f"U+{cp:04X}" for cp in sorted(missing_everywhere)[:64]],
        }
        return BackendResult(
            candidate_dirs=tuple(path.resolve() for path in self._candidate_dirs),
            metadata=metadata,
        )
