from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..image_cache import read_gray_u8
from ..proxy import ink_bbox
from ..util import ensure_dir
from .base import (
    CANDIDATE_SIZE,
    BackendRequest,
    BackendResult,
    BackendUnavailable,
    CandidateGeometry,
    candidate_filename,
    codepoint_from_filename,
)


CONTENT_SIZE = 256
STYLE_SIZE = 128

# zi2zi-JiT's README recommends this combination as the current fast setting
# for JiT-B/16.  cfg 2.4 is recommended for JiT-L/16.
DEFAULT_SAMPLING_METHOD = "ab2"
DEFAULT_SAMPLING_STEPS = 20
DEFAULT_CFG_SCALE = 2.6

# One chunk of 2048 glyphs is roughly a 400 MB content array before
# compression.  A whole 30k-glyph font would be about 6 GB, which neither fits
# comfortably in memory nor survives np.savez_compressed, so generation is
# always chunked.
DEFAULT_CHUNK_SIZE = 2048


def _ink_at_size(path: str | Path, size: int) -> np.ndarray:
    gray = np.asarray(read_gray_u8(path), dtype=np.float32)
    ink = (1.0 - gray / 255.0).clip(0.0, 1.0)
    if ink.shape != (int(size), int(size)):
        ink = cv2.resize(ink, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    return ink


def _probe_source(checkpoint: Path) -> str:
    return (
        "import json,sys,torch\n"
        f"ck=torch.load(r{str(checkpoint)!r},map_location='cpu',weights_only=False)\n"
        "a=ck['args']\n"
        "print('HSF_PROBE '+json.dumps({\n"
        "  'model':getattr(a,'model',''),\n"
        "  'img_size':int(getattr(a,'img_size',256)),\n"
        "  'num_fonts':int(getattr(a,'num_fonts',0)),\n"
        "  'num_chars':int(getattr(a,'num_chars',0)),\n"
        "  'keys':[k for k in ck if k!='args'],\n"
        "}))\n"
    )


class Zi2ziJitBackend:
    """Generate glyphs by driving a local zi2zi-JiT checkout as a subprocess.

    zi2zi-JiT is invoked rather than imported. ``generate_chars.py`` globally
    monkey-patches ``torch.Tensor.cuda``, ``torch.nn.Module.cuda``,
    ``torch.compile`` and ``torch.cuda.amp.autocast``, and calls
    ``misc.init_distributed_mode``; importing it would corrupt this project's
    own torch usage in the same process. A subprocess also releases VRAM
    cleanly before the refiner and QA stages run.

    The interpreter may be the same one running HanziStyleForge: zi2zi-JiT's
    inference path needs only torch, numpy, cv2 and einops, so its pinned
    ``numpy=1.22`` / ``torch==2.5.1`` environment is not actually required.
    """

    name = "zi2zi-jit"

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        *,
        python_executable: str | Path | None = None,
        model: str | None = None,
        sampling_method: str = DEFAULT_SAMPLING_METHOD,
        num_sampling_steps: int = DEFAULT_SAMPLING_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        batch_size: int = 16,
        font_label: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        style_pool_size: int = 64,
        timeout_seconds: int = 0,
        keep_intermediate: bool = False,
        disable_torch_compile: bool = True,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.checkpoint = Path(checkpoint)
        self.python_executable = Path(python_executable) if python_executable else Path(sys.executable)
        self.model = model
        self.sampling_method = str(sampling_method)
        self.num_sampling_steps = int(num_sampling_steps)
        self.cfg_scale = float(cfg_scale)
        self.batch_size = max(1, int(batch_size))
        self.font_label = font_label
        self.chunk_size = max(1, int(chunk_size))
        self.style_pool_size = max(1, int(style_pool_size))
        self.timeout_seconds = max(0, int(timeout_seconds))
        self.keep_intermediate = bool(keep_intermediate)
        self.disable_torch_compile = bool(disable_torch_compile)
        self._checkpoint_info: dict[str, Any] | None = None

    # ------------------------------------------------------------------ setup

    def _run(self, arguments: Sequence[str], *, description: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        # generate_chars.py enters distributed mode whenever these are present.
        # A stray value inherited from an unrelated launcher would make the
        # subprocess wait for peers that never arrive.
        for variable in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            environment.pop(variable, None)
        environment["PYTHONIOENCODING"] = "utf-8"
        if self.disable_torch_compile:
            # model_jit.py decorates two forward() methods with @torch.compile at
            # class definition time. generate_chars.py neutralizes torch.compile
            # only for non-CUDA devices, so on CUDA the decorators stay live and
            # inductor demands Triton, which has no official Windows build. This
            # project is Windows-first, so dynamo is disabled by default; set
            # disable_torch_compile=False if you have installed Triton and want
            # the compiled speedup.
            environment["TORCHDYNAMO_DISABLE"] = "1"
        try:
            return subprocess.run(
                [str(self.python_executable), *arguments],
                cwd=str(self.repo_dir),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds or None,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailable(
                f"zi2zi-JiT {description} exceeded the configured timeout of "
                f"{self.timeout_seconds}s. Raise backend.zi2zi_jit.timeout_seconds or set it to 0 "
                "to wait indefinitely."
            ) from exc
        except OSError as exc:
            raise BackendUnavailable(
                f"Could not start the zi2zi-JiT {description} subprocess using "
                f"{self.python_executable}: {exc}"
            ) from exc

    def checkpoint_info(self) -> dict[str, Any]:
        """Read model metadata from the checkpoint, loading it at most once.

        This doubles as the real environment check: it proves the checkpoint
        unpickles and that zi2zi-JiT imports cleanly in the configured
        interpreter, which is exactly what would otherwise fail hours into a
        run.
        """

        if self._checkpoint_info is not None:
            return self._checkpoint_info
        completed = self._run(["-c", _probe_source(self.checkpoint)], description="checkpoint probe")
        marker = "HSF_PROBE "
        payload = ""
        for line in (completed.stdout or "").splitlines():
            if line.startswith(marker):
                payload = line[len(marker):]
        if completed.returncode != 0 or not payload:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            raise BackendUnavailable(
                "Could not read the zi2zi-JiT checkpoint "
                f"{self.checkpoint}.\nInterpreter: {self.python_executable}\n"
                "The inference path needs torch, numpy, opencv and einops; install any that are "
                f"missing.\nSubprocess output:\n{detail}"
            )
        self._checkpoint_info = json.loads(payload)
        return self._checkpoint_info

    def preflight(self) -> None:
        if not self.repo_dir.is_dir():
            raise BackendUnavailable(
                f"The zi2zi-JiT repository was not found at {self.repo_dir}.\n"
                "Clone it with:  git clone https://github.com/kaonashi-tyc/zi2zi-JiT\n"
                "then set backend.zi2zi_jit.repo_dir to that directory."
            )
        script = self.repo_dir / "generate_chars.py"
        if not script.is_file():
            raise BackendUnavailable(
                f"{self.repo_dir} does not look like a zi2zi-JiT checkout: generate_chars.py is missing.\n"
                "Point backend.zi2zi_jit.repo_dir at the repository root."
            )
        if not self.checkpoint.is_file():
            raise BackendUnavailable(
                f"The zi2zi-JiT checkpoint was not found at {self.checkpoint}.\n"
                "Download JiT-B/16 or JiT-L/16 from the Google Drive link in the zi2zi-JiT README, "
                "then set backend.zi2zi_jit.checkpoint to the .pth file."
            )
        if not self.python_executable.is_file():
            raise BackendUnavailable(
                f"The configured interpreter {self.python_executable} does not exist.\n"
                "Set backend.zi2zi_jit.python_executable to a python.exe that can import torch, "
                "or leave it empty to reuse the interpreter running HanziStyleForge."
            )
        self.checkpoint_info()

    def resolved_font_label(self) -> int:
        """Return the font embedding index to condition on.

        The published checkpoints were trained on 400+ fonts that do not
        include the user's target font, and ``LabelEmbedder`` holds a
        ``num_fonts + 1`` row table whose final row is the label-drop token.
        Conditioning on any real training font would inject that font's
        identity and fight the style reference images, so the default is the
        drop token: font identity unconditional, style supplied entirely by the
        reference images. Training used this configuration for a large share of
        its steps, so it is in distribution.

        After a LoRA fine-tune the target font occupies a real index (0 for a
        dataset laid out as ``001_<name>``) and ``font_label`` should be set.
        """

        if self.font_label is not None:
            return int(self.font_label)
        return int(self.checkpoint_info()["num_fonts"])

    # ------------------------------------------------------------------- data

    def _glyph_renderer(self, font: str | Path, resolution: int):
        """Instantiate zi2zi-JiT's own rasterizer for a font.

        The class is imported from the configured checkout rather than
        reimplemented, because the fine-tuning dataset was rasterized by this
        exact code. Any divergence would shift the content distribution the
        model was tuned on and degrade output silently. Only
        ``data_processing.font_utils`` is imported, which pulls in fontTools and
        PIL but no torch, so none of the monkey-patching that forces the
        subprocess boundary applies here.
        """

        repo = str(self.repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from data_processing.font_utils import GlyphRenderer
        except ImportError as exc:
            raise BackendUnavailable(
                f"Could not import data_processing.font_utils from {self.repo_dir}: {exc}\n"
                "Point backend.zi2zi_jit.repo_dir at a complete zi2zi-JiT checkout."
            ) from exc
        try:
            return GlyphRenderer(str(font), int(resolution))
        except Exception as exc:
            raise BackendUnavailable(
                f"zi2zi-JiT could not rasterize {font}: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _to_chw(image: Any, size: int) -> np.ndarray:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape[:2] != (size, size):
            array = cv2.resize(array, (size, size), interpolation=cv2.INTER_LANCZOS4)
        return array.transpose(2, 0, 1)

    @staticmethod
    def _rgb_uint8(path: str | Path, size: int) -> np.ndarray:
        gray = np.asarray(read_gray_u8(path), dtype=np.uint8)
        if gray.shape != (size, size):
            interpolation = cv2.INTER_AREA if gray.shape[0] > size else cv2.INTER_LANCZOS4
            gray = cv2.resize(gray, (size, size), interpolation=interpolation)
        return np.repeat(gray[None, :, :], 3, axis=0)

    def measure_geometry(
        self,
        request: BackendRequest,
        *,
        target_size: int,
        sample_count: int = 48,
    ) -> CandidateGeometry:
        """Estimate the transform from zi2zi-JiT's frame into this project's.

        Both renderers draw the same reference outlines, so comparing ink
        bounding boxes of the same glyphs rasterized both ways recovers the
        uniform scale and translation between the frames. Medians over many
        glyphs are used because a single glyph's box is sensitive to overshoot
        and to strokes that touch the frame edge.
        """

        if request.reference_font is None:
            raise BackendUnavailable(
                "Measuring backend geometry needs the reference font; BackendRequest.reference_font is unset."
            )
        renderer = self._glyph_renderer(request.reference_font, CONTENT_SIZE)
        scales: list[float] = []
        centres: list[tuple[float, float, float, float]] = []
        for glyph in request.glyphs[: max(8, int(sample_count))]:
            image = renderer.render(glyph.codepoint)
            if image is None:
                continue
            foreign = 1.0 - np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            ours = _ink_at_size(glyph.ref_png, target_size)
            foreign_box = ink_bbox(foreign >= 0.5)
            ours_box = ink_bbox(ours >= 0.5)
            if foreign_box is None or ours_box is None:
                continue
            fx0, fy0, fx1, fy1 = foreign_box
            ox0, oy0, ox1, oy1 = ours_box
            foreign_extent = max(fx1 - fx0, fy1 - fy0)
            ours_extent = max(ox1 - ox0, oy1 - oy0)
            if foreign_extent < 4 or ours_extent < 4:
                continue
            scales.append(ours_extent / float(foreign_extent))
            centres.append((
                (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0,
                (ox0 + ox1) / 2.0, (oy0 + oy1) / 2.0,
            ))
        if not scales:
            raise BackendUnavailable(
                "Could not measure backend geometry: no sampled glyph rendered in both frames."
            )
        scale = float(np.median(scales))
        offsets_x = [ocx - scale * fcx for fcx, _, ocx, _ in centres]
        offsets_y = [ocy - scale * fcy for _, fcy, _, ocy in centres]
        return CandidateGeometry(
            scale=scale,
            offset_x=float(np.median(offsets_x)),
            offset_y=float(np.median(offsets_y)),
        )

    def _style_codepoints(self, request: BackendRequest) -> list[int]:
        available = sorted(request.style_glyph_pngs)
        if not available:
            return []
        if len(available) <= self.style_pool_size:
            return available
        positions = np.linspace(0, len(available) - 1, self.style_pool_size, dtype=np.int64)
        return [available[int(index)] for index in positions]

    def _style_pool(self, request: BackendRequest) -> list[Path]:
        available = sorted(request.style_glyph_pngs.items())
        if not available:
            raise BackendUnavailable(
                "The zi2zi-JiT backend needs rendered target glyphs to use as style references, "
                "but none were supplied. Run prepare first so the target render cache exists."
            )
        if len(available) <= self.style_pool_size:
            return [path for _, path in available]
        # Even coverage of the ordered Unicode range keeps the pool
        # deterministic and structurally varied rather than clustered.
        positions = np.linspace(0, len(available) - 1, self.style_pool_size, dtype=np.int64)
        return [available[int(index)][1] for index in positions]

    def _write_chunk_npz(
        self,
        path: Path,
        glyphs: Sequence[Any],
        style_pool: Sequence[Path],
        font_label: int,
        candidate_index: int,
        content_renderer: Any = None,
        style_renderer: Any = None,
        style_codepoints: Sequence[int] = (),
    ) -> None:
        count = len(glyphs)
        content = np.empty((count, 3, CONTENT_SIZE, CONTENT_SIZE), dtype=np.uint8)
        style = np.empty((count, 3, STYLE_SIZE, STYLE_SIZE), dtype=np.uint8)
        unicode_labels = np.empty(count, dtype=np.int64)
        for index, glyph in enumerate(glyphs):
            rendered = content_renderer.render(glyph.codepoint) if content_renderer is not None else None
            if rendered is not None:
                # Rasterized by zi2zi-JiT's own renderer so the content matches
                # what the fine-tuning dataset was built from.
                content[index] = self._to_chw(rendered, CONTENT_SIZE)
            else:
                content[index] = self._rgb_uint8(glyph.ref_png, CONTENT_SIZE)
            # Training drew a random reference per sample, so varying the
            # reference per glyph matches the training distribution.  Seeding it
            # on (candidate, codepoint) keeps a rerun reproducible while giving
            # each candidate a different style view.
            styled = None
            if style_renderer is not None and style_codepoints:
                choice = (glyph.codepoint * 2654435761 + candidate_index * 40503) % len(style_codepoints)
                # Training rendered style references at CONTENT_SIZE and then
                # downscaled them into the reference grid, so reproduce both
                # steps rather than rasterizing straight to STYLE_SIZE.
                styled = style_renderer.render(int(style_codepoints[choice]))
            if styled is not None:
                style[index] = self._to_chw(styled, STYLE_SIZE)
            else:
                choice = (glyph.codepoint * 2654435761 + candidate_index * 40503) % len(style_pool)
                style[index] = self._rgb_uint8(style_pool[choice], STYLE_SIZE)
            unicode_labels[index] = int(glyph.codepoint)
        np.savez_compressed(
            path,
            font_labels=np.full(count, int(font_label), dtype=np.int64),
            # char_labels is unpacked by LabelEmbedder.encode and never used;
            # the model has no character embedding. Zeros are correct.
            char_labels=np.zeros(count, dtype=np.int64),
            unicode_labels=unicode_labels,
            content_images=content,
            style_images=style,
            num_original_samples=np.int64(count),
        )

    # -------------------------------------------------------------- execution

    def _generated_dir(self, output_dir: Path) -> Path:
        """Locate the run directory generate_chars.py composed for itself."""

        candidates = sorted(output_dir.glob("*/generated"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise BackendUnavailable(
                f"zi2zi-JiT reported success but produced no output under {output_dir}."
            )
        raise BackendUnavailable(
            f"{output_dir} contains {len(candidates)} zi2zi-JiT run directories; expected one. "
            "Delete the stale ones and rerun."
        )

    def _run_chunk(self, npz_path: Path, output_dir: Path) -> Path:
        arguments = [
            "generate_chars.py",
            "--checkpoint", str(self.checkpoint),
            "--test_npz", str(npz_path),
            "--output_dir", str(output_dir),
            "--batch_size", str(self.batch_size),
            "--sampling_method", self.sampling_method,
            "--num_sampling_steps", str(self.num_sampling_steps),
            "--cfg", str(self.cfg_scale),
        ]
        if self.model:
            arguments += ["--model", self.model]
        completed = self._run(arguments, description="generation")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
            raise BackendUnavailable(
                f"zi2zi-JiT generation failed (exit code {completed.returncode}).\n"
                f"Command: generate_chars.py --test_npz {npz_path.name}\n"
                f"Output:\n{detail}"
            )
        return self._generated_dir(output_dir)

    def generate(self, request: BackendRequest) -> BackendResult:
        self.preflight()
        font_label = self.resolved_font_label()
        style_pool = self._style_pool(request)
        candidate_count = max(1, int(request.candidate_count))
        root = ensure_dir(request.output_root)
        scratch = Path(tempfile.mkdtemp(prefix="zi2zi-jit-", dir=str(root)))

        content_renderer = None
        style_renderer = None
        style_codepoints: list[int] = []
        geometry: CandidateGeometry | None = None
        if request.reference_font is not None:
            content_renderer = self._glyph_renderer(request.reference_font, CONTENT_SIZE)
            geometry = self.measure_geometry(request, target_size=CANDIDATE_SIZE * 2)
            # Style references must come from the same rasterizer as the
            # content, otherwise the style encoder sees a frame it never met
            # during fine-tuning.
            style_renderer = self._glyph_renderer(request.style_font, CONTENT_SIZE)
            style_codepoints = self._style_codepoints(request)

        candidate_dirs: list[Path] = []
        chunk_reports: list[dict[str, Any]] = []
        try:
            for candidate_index in range(candidate_count):
                target = ensure_dir(root / f"candidate_{candidate_index:02d}")
                candidate_dirs.append(target)
                for start in range(0, len(request.glyphs), self.chunk_size):
                    chunk = request.glyphs[start:start + self.chunk_size]
                    outstanding = [
                        glyph for glyph in chunk
                        if not (target / candidate_filename(glyph.codepoint)).is_file()
                    ]
                    if not outstanding:
                        # Already produced by an earlier interrupted run.
                        continue
                    label = f"c{candidate_index:02d}_s{start:07d}"
                    npz_path = scratch / f"{label}.npz"
                    output_dir = scratch / label
                    self._write_chunk_npz(
                        npz_path, outstanding, style_pool, font_label, candidate_index,
                        content_renderer=content_renderer,
                        style_renderer=style_renderer,
                        style_codepoints=style_codepoints,
                    )
                    produced = self._run_chunk(npz_path, output_dir)
                    collected = 0
                    for image in sorted(produced.iterdir()):
                        codepoint = codepoint_from_filename(image.name)
                        if codepoint is None:
                            continue
                        shutil.copyfile(image, target / candidate_filename(codepoint))
                        collected += 1
                    chunk_reports.append(
                        {
                            "candidate": candidate_index,
                            "offset": start,
                            "requested": len(outstanding),
                            "collected": collected,
                        }
                    )
                    if not self.keep_intermediate:
                        npz_path.unlink(missing_ok=True)
                        shutil.rmtree(output_dir, ignore_errors=True)
        finally:
            if not self.keep_intermediate:
                shutil.rmtree(scratch, ignore_errors=True)

        requested = set(request.codepoints())
        covered: set[int] = set()
        for directory in candidate_dirs:
            covered |= {
                codepoint for codepoint in requested
                if (directory / candidate_filename(codepoint)).is_file()
            }
        info = self.checkpoint_info()
        metadata: dict[str, Any] = {
            "backend": self.name,
            "checkpoint": str(self.checkpoint.resolve()),
            "model": info.get("model", ""),
            "interpreter": str(self.python_executable),
            "font_label": font_label,
            "font_label_is_unconditional": font_label == int(info["num_fonts"]),
            "sampling_method": self.sampling_method,
            "num_sampling_steps": self.num_sampling_steps,
            "cfg_scale": self.cfg_scale,
            "candidate_count": candidate_count,
            "chunk_size": self.chunk_size,
            "style_pool_size": len(style_pool),
            "requested_glyphs": len(requested),
            "missing_from_all_candidates": len(requested - covered),
            "chunks": chunk_reports,
            # Present when content was rasterized by zi2zi-JiT's renderer; the
            # caller must apply it, otherwise every outline lands in the wrong
            # place inside the em box.
            "content_renderer": "zi2zi-JiT" if content_renderer is not None else "hanzistyleforge",
            "geometry": None if geometry is None else geometry.as_dict(),
            "geometry_target_size": CANDIDATE_SIZE * 2,
        }
        return BackendResult(
            candidate_dirs=tuple(path.resolve() for path in candidate_dirs),
            metadata=metadata,
        )
