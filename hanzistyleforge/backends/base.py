from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import cv2
import numpy as np

from ..image_cache import read_gray_u8
from ..proxy import ink_bbox


# The contract filename.  ``U+4E00.png`` is used rather than the internal
# ``cp_filename`` form (``U4E00.png``) because it is what zi2zi-JiT emits and
# what a human assembling a directory by hand will naturally write.  The
# adapter layer translates to the internal naming, so backends never need to
# know about it.
_CODEPOINT_PATTERN = re.compile(r"U\+([0-9A-Fa-f]{4,6})")

CANDIDATE_SIZE = 256
"""Backend output resolution.  Matches zi2zi-JiT's native 256x256."""

SUPPORTED_SUFFIXES = (".png", ".PNG")


class BackendUnavailable(RuntimeError):
    """Raised by ``preflight`` when a backend cannot run in this environment.

    The message must name the missing thing and the action that fixes it.  A
    multi-week run should never discover a missing checkpoint after generating
    twenty thousand glyphs, which is why ``preflight`` is separate from
    ``generate`` and is always called first.
    """


@dataclass(frozen=True)
class GlyphRequest:
    """One glyph to generate.

    ``ref_png`` and ``ref_proxy_png`` are the caches produced by ``prepare``;
    they are rendered from ``refs/ref.otf`` and are the only structural source
    a backend may read.  No backend receives the target glyph for the same
    codepoint, which keeps the project's data-flow contract intact: style comes
    from the target font as a whole, structure comes from the reference.
    """

    codepoint: int
    ref_png: Path
    ref_proxy_png: Path


@dataclass(frozen=True)
class BackendRequest:
    """Everything a backend needs for one generation pass."""

    glyphs: tuple[GlyphRequest, ...]
    style_font: Path
    style_glyph_pngs: Mapping[int, Path]
    candidate_count: int
    output_root: Path
    work_dir: Path

    def codepoints(self) -> tuple[int, ...]:
        return tuple(item.codepoint for item in self.glyphs)


@dataclass(frozen=True)
class BackendResult:
    """One directory per candidate.

    ``candidate_dirs`` may be shorter than the requested ``candidate_count``
    and individual directories may be missing glyphs.  Both cases are normal:
    the caller treats every produced image as one more candidate for the
    existing selection logic and falls back to the reference-derived candidate
    whenever a glyph has none.  A backend must never raise merely because one
    glyph failed.
    """

    candidate_dirs: tuple[Path, ...]
    metadata: dict[str, Any]


@runtime_checkable
class GenerationBackend(Protocol):
    """Contract for a pluggable generation backend.

    Implementations produce ``CANDIDATE_SIZE`` x ``CANDIDATE_SIZE`` grayscale
    PNGs named ``U+XXXX.png``, white background and black ink, one directory
    per candidate.
    """

    name: str

    def preflight(self) -> None:
        """Fail fast with an actionable message if the backend cannot run."""

    def generate(self, request: BackendRequest) -> BackendResult:
        """Produce candidate directories for ``request``."""


def candidate_filename(codepoint: int) -> str:
    """Return the contract filename for a codepoint.

    ``:04X`` is a minimum width, not a fixed one, so U+4E00 stays four digits
    while U+20000 renders as five.  This matches zi2zi-JiT's own formatter
    (``data_processing/pipeline.py::_format_codepoint`` and the filename built
    in ``generate_chars.py``) exactly, so names round-trip between the two
    projects without a translation table.
    """

    return f"U+{int(codepoint):04X}.png"


def codepoint_from_filename(name: str | Path) -> int | None:
    """Extract a codepoint from a candidate filename.

    Accepts the contract form ``U+4E00.png`` and also zi2zi-JiT's native
    ``0000_U+4E00.png``, so a raw zi2zi-JiT output directory can be consumed by
    the ``dir`` backend without renaming anything first.
    """

    match = _CODEPOINT_PATTERN.search(Path(name).stem)
    if match is None:
        return None
    try:
        return int(match.group(1), 16)
    except ValueError:
        return None


def read_candidate_dir(directory: str | Path) -> dict[int, Path]:
    """Map codepoint to image path for one candidate directory.

    Files whose name carries no codepoint are ignored rather than treated as an
    error, so a stray ``metadata.json`` or contact sheet does not break a run.
    A codepoint appearing twice is a genuine ambiguity and raises.
    """

    root = Path(directory)
    result: dict[int, Path] = {}
    for path in sorted(root.iterdir() if root.is_dir() else []):
        if not path.is_file() or path.suffix not in SUPPORTED_SUFFIXES:
            continue
        codepoint = codepoint_from_filename(path.name)
        if codepoint is None:
            continue
        if codepoint in result:
            raise ValueError(
                f"{root} maps U+{codepoint:04X} to both {result[codepoint].name} and {path.name}; "
                "remove the duplicate so candidate selection is unambiguous."
            )
        result[codepoint] = path
    return result


def _ink_from_png(path: str | Path) -> np.ndarray:
    gray = np.asarray(read_gray_u8(path), dtype=np.float32)
    return (1.0 - gray / 255.0).clip(0.0, 1.0)


def _fit_to_reference_bbox(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Affine-fit the candidate ink bounding box onto the reference one.

    Used only for images of unknown provenance.  It normalizes position and
    scale but deliberately preserves the candidate's own stroke weight, so a
    heavier or lighter style survives the transform.
    """

    source = ink_bbox(candidate >= 0.5)
    destination = ink_bbox(reference >= 0.5)
    if source is None or destination is None:
        return candidate
    sx0, sy0, sx1, sy1 = source
    dx0, dy0, dx1, dy1 = destination
    source_w, source_h = max(1, sx1 - sx0), max(1, sy1 - sy0)
    scale_x = (dx1 - dx0) / float(source_w)
    scale_y = (dy1 - dy0) / float(source_h)
    matrix = np.asarray(
        [
            [scale_x, 0.0, dx0 - sx0 * scale_x],
            [0.0, scale_y, dy0 - sy0 * scale_y],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        candidate.astype(np.float32),
        matrix,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return warped.clip(0.0, 1.0)


def normalize_candidate(
    candidate_png: str | Path,
    *,
    target_size: int,
    mode: str = "resample",
    reference_png: str | Path | None = None,
) -> np.ndarray:
    """Convert one backend PNG into an ink array in the project's geometry.

    Backends emit 256x256 images while the rest of the program works at
    ``render.size`` (512 by default) inside an em box derived from the target
    font's vertical metrics.  ``build_font`` passes ``render.pad`` relative to
    that size, so feeding a raw 256px image downstream would misplace every
    outline.  This function is the single place that reconciles the two.

    ``resample`` is correct when the backend was fed content images rendered by
    this project's own ``FontRenderer``: the model then works in our geometry
    already and only the resolution differs.  ``ref_bbox_fit`` re-registers an
    image whose geometry is unknown, which is what a hand-assembled directory
    needs.
    """

    ink = _ink_from_png(candidate_png)
    size = int(target_size)
    if ink.shape != (size, size):
        # INTER_AREA for downscaling preserves stroke mass; LANCZOS4 keeps
        # edges crisp when upsampling 256 -> 512.
        interpolation = cv2.INTER_AREA if ink.shape[0] > size else cv2.INTER_LANCZOS4
        ink = cv2.resize(ink, (size, size), interpolation=interpolation).clip(0.0, 1.0)

    normalized = str(mode).lower()
    if normalized == "resample":
        return ink.astype(np.float32)
    if normalized != "ref_bbox_fit":
        raise ValueError(
            f"Unknown backend normalization mode {mode!r}; expected 'resample' or 'ref_bbox_fit'."
        )
    if reference_png is None:
        raise ValueError("normalization mode 'ref_bbox_fit' requires reference_png.")
    reference = _ink_from_png(reference_png)
    if reference.shape != (size, size):
        reference = cv2.resize(reference, (size, size), interpolation=cv2.INTER_AREA).clip(0.0, 1.0)
    return _fit_to_reference_bbox(ink, reference).astype(np.float32)
