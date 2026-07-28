from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .backends import BackendRequest, BackendUnavailable, GlyphRequest, read_candidate_dir
from .backends.base import CandidateGeometry, normalize_candidate
from .backends.factory import backend_name, create_backend
from .contract import validate_data_flow_contract
from .fusion_training import FUSION_CHECKPOINT_VERSION
from .inference import (
    SELECTION_FIELDS,
    _confidence,
    _emergency_fallback_row,
    _evaluate_family,
    _style_profile_for_complexity,
    _threshold_candidates,
)
from .longrun import LongRunGuard
from .proxy import make_reference_fallbacks, read_ink, read_proxy, save_ink
from .topology import topology_signature
from .util import (
    cp_filename,
    deep_merge,
    ensure_dir,
    load_json,
    read_csv,
    save_codepoints,
    save_json,
    sha256_file,
    write_csv,
)


BACKEND_SELECTION_FIELDS = SELECTION_FIELDS + [
    "backend_name",
    "backend_candidate_count",
    "backend_candidate_index",
    "backend_disagreement",
]


def _style_glyph_pngs(work: Path, limit: int) -> dict[int, Path]:
    """Collect rendered target glyphs for backends that need style references.

    The style source is the target font only, matching the project's data-flow
    contract: reference renders never enter this mapping.
    """

    style_csv = work / "audit" / "style_source.csv"
    if not style_csv.is_file():
        return {}
    rows = [row for row in read_csv(style_csv) if int(row.get("trainable", 0) or 0)]
    if not rows:
        return {}
    if len(rows) > limit > 0:
        positions = np.linspace(0, len(rows) - 1, int(limit), dtype=np.int64)
        rows = [rows[int(index)] for index in positions]
    return {int(row["codepoint"]): Path(row["target_path"]) for row in rows}


def _backend_fingerprint(cfg: dict[str, Any], analysis_path: Path, name: str) -> str:
    # selection_workers only changes how the work is distributed, never the
    # result, so it must not invalidate a partially completed run.
    backend = {
        key: value for key, value in cfg.get("backend", {}).items()
        if key != "selection_workers"
    }
    payload = {
        "version": FUSION_CHECKPOINT_VERSION,
        "analysis": sha256_file(analysis_path),
        "backend": backend,
        "backend_name": name,
        "topology": cfg.get("topology", {}),
        "render": cfg.get("render", {}),
        "inference": cfg.get("inference", {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_WORKER_CONTEXT: dict[str, Any] = {}


def _init_selection_worker(context: dict[str, Any]) -> None:
    """Install the per-run constants in a worker process.

    Passed once per worker rather than once per glyph: the style profiles and
    topology configuration are identical for every glyph, and pickling them
    tens of thousands of times would cost more than the work itself.
    """

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    # Each worker is one core's worth of work. Letting OpenCV start its own
    # pool inside every worker oversubscribes the machine and runs slower than
    # the single-threaded version.
    cv2.setNumThreads(1)


def _select_glyph(row: dict[str, str], images: list[str], context: dict[str, Any]) -> dict[str, Any]:
    """Score one glyph's backend candidates and pick a winner.

    Pure with respect to the parent process: everything it needs arrives in
    ``context`` or is read from disk, and the only thing it writes is that
    glyph's own chosen image, so parallel workers never contend.
    """

    cp = int(row["codepoint"])
    name = context["name"]
    inference_size = context["inference_size"]
    analysis_size = context["analysis_size"]
    profile_size = context["profile_size"]
    topology_cfg = context["topology_cfg"]
    threshold_offsets = context["threshold_offsets"]
    weights = context["weights"]
    maximum_border = context["maximum_border"]
    chosen_dir = Path(context["chosen_dir"])

    try:
        proxy_metrics = read_proxy(row["ref_proxy_path"])
        reference_ink = read_ink(row["ref_path"])
        if reference_ink.shape != (inference_size, inference_size):
            reference_ink = cv2.resize(
                reference_ink, (inference_size, inference_size), interpolation=cv2.INTER_AREA
            )
        profile = _style_profile_for_complexity(
            context["style_profiles"], float(row.get("complexity", 0.0) or 0.0)
        )
        reference_signature = topology_signature(
            reference_ink,
            size=int(topology_cfg.get("analysis_size", analysis_size)),
            prune_iterations=int(topology_cfg.get("prune_iterations", 1)),
        )

        families: dict[str, dict[str, Any]] = {}
        probabilities: list[np.ndarray] = []
        for index, image in enumerate(images):
            probability = normalize_candidate(
                image,
                target_size=inference_size,
                mode=context["normalization"],
                reference_png=row["ref_path"],
                geometry=context["geometry"],
            )
            probabilities.append(probability)
            # Backend output is antialiased, so sweeping thresholds over it
            # produces genuinely different masks rather than copies.
            label = f"{name}_c{index:02d}"
            families[label] = _evaluate_family(
                label,
                _threshold_candidates(probability, 0.5, threshold_offsets, label),
                proxy_metrics, reference_ink, profile, analysis_size, profile_size,
                maximum_border, topology_cfg, weights["backend"], reference_signature,
            )

        if len(probabilities) >= 2:
            stack = np.stack(probabilities, axis=0)
            for suffix, blended in (("mean", stack.mean(axis=0)), ("median", np.median(stack, axis=0))):
                label = f"{name}_{suffix}"
                families[label] = _evaluate_family(
                    label,
                    _threshold_candidates(blended, 0.5, threshold_offsets, label),
                    proxy_metrics, reference_ink, profile, analysis_size, profile_size,
                    maximum_border, topology_cfg, weights["backend"], reference_signature,
                )
            disagreement = float(stack.std(axis=0).mean())
        else:
            disagreement = 0.0

        families["fallback"] = _evaluate_family(
            "fallback",
            make_reference_fallbacks(reference_ink, profile, threshold=context["render_threshold"]),
            proxy_metrics, reference_ink, profile, analysis_size, profile_size,
            maximum_border, topology_cfg, weights["fallback"], reference_signature,
        )

        consensus = np.mean(np.stack(probabilities, axis=0), axis=0) if probabilities else None
        for source, family in families.items():
            is_fallback = source == "fallback"
            family["confidence"] = _confidence(
                family,
                None if is_fallback else consensus,
                context["keep_threshold"],
                gate_relaxation=1.0 if is_fallback else context["gate_relaxation"],
                delta_relaxation=1.0 if is_fallback else context["delta_relaxation"],
            )

        # The reference fallback is a safety net here, not a competitor. Every
        # score in _evaluate_family measures similarity to the reference
        # structure, and the fallback is derived from that structure, so it wins
        # any ranking it takes part in. The native path can afford to rank it
        # alongside its own candidates because those are structure-locked to the
        # same reference; a backend doing style transfer deviates by design, and
        # letting the fallback compete would silently rebuild the font out of
        # reference outlines. It is used only when nothing else clears the gate.
        backend_families = [family for source, family in families.items() if source != "fallback"]
        passing = [family for family in backend_families if family["validation"]["hard_pass"]]
        if passing:
            chosen = min(passing, key=lambda item: (float(item["total_score"]), -float(item["confidence"])))
        else:
            chosen = families["fallback"]
        chosen_source = str(chosen["source"])
        chosen_path = chosen_dir / cp_filename(cp)
        save_ink(chosen_path, np.asarray(chosen["mask"], dtype=np.float32).clip(0.0, 1.0))

        has_target = bool(int(row.get("has_target", 0)))
        notes = ""
        if not chosen["validation"]["hard_pass"]:
            notes = "All backend candidates failed the hard topology gate; ref-derived fallback selected."
        elif not probabilities:
            notes = f"Backend {name} produced no image for this glyph."

        record: dict[str, Any] = {field: "" for field in BACKEND_SELECTION_FIELDS}
        record.update({
            "codepoint": cp,
            "unicode": row.get("unicode", f"U+{cp:04X}"),
            "char": row.get("char", chr(cp)),
            "has_target": int(has_target),
            "locl_sensitive": int(row.get("locl_sensitive", 0)),
            "preliminary_status": row.get("preliminary_status", "rebuild"),
            "final_action": "replace" if has_target else "add",
            "chosen_source": chosen_source,
            "chosen_label": chosen["label"],
            "chosen_path": str(chosen_path.resolve()),
            "ref_path": row.get("ref_path", ""),
            "target_path": row.get("target_path", ""),
            "ref_proxy_path": row.get("ref_proxy_path", ""),
            "chosen_structure_score": chosen["structure"]["structure_score"],
            "chosen_topology_score": chosen["topology"]["topology_score"],
            "chosen_topology_pass": int(chosen["validation"]["hard_pass"]),
            "chosen_component_delta": chosen["topology"]["component_delta"],
            "chosen_hole_delta": chosen["topology"]["hole_delta"],
            "chosen_endpoint_delta": chosen["topology"]["endpoint_delta"],
            "chosen_junction_delta": chosen["topology"]["junction_delta"],
            "chosen_confidence": chosen["confidence"],
            "pseudo_eligible": 0,
            "rejection_reasons": str({
                source: family["validation"]["reasons"]
                for source, family in families.items()
                if not family["validation"]["hard_pass"]
            }),
            "notes": notes,
            # Columns the existing QA report reads.
            "neural_structure_score": chosen["structure"]["structure_score"],
            "neural_topology_score": chosen["topology"]["topology_score"],
            "neural_style_score": chosen.get("style_score", ""),
            "neural_confidence": chosen["confidence"],
            "fallback_structure_score": families["fallback"]["structure"]["structure_score"],
            "fallback_topology_score": families["fallback"]["topology"]["topology_score"],
            "backend_name": name,
            "backend_candidate_count": len(probabilities),
            "backend_candidate_index": chosen_source,
            "backend_disagreement": disagreement,
        })
        return record
    except Exception as exc:
        emergency = _emergency_fallback_row(row, cp, chosen_dir, exc)
        record = {field: "" for field in BACKEND_SELECTION_FIELDS}
        record.update(emergency)
        record["backend_name"] = name
        record["backend_candidate_count"] = 0
        return record


def _select_glyph_task(task: tuple[dict[str, str], list[str]]) -> dict[str, Any]:
    return _select_glyph(task[0], task[1], _WORKER_CONTEXT)


def _resolve_worker_count(configured: int) -> int:
    """Pick a worker count. 0 means auto."""

    requested = int(configured)
    if requested > 0:
        return requested
    available = os.cpu_count() or 1
    # Half the cores, capped: each worker holds several 512x512 float buffers,
    # and the stage is memory-bandwidth bound well before it is core bound.
    return max(1, min(8, available // 2))


def _write_partial(path: Path, state: Path, rows: list[dict[str, Any]], fingerprint: str, total: int) -> None:
    write_csv(path, rows, BACKEND_SELECTION_FIELDS)
    save_json(state, {
        "version": FUSION_CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "completed_count": len(rows),
        "target_count": int(total),
    })


def generate_with_backend(
    cfg: dict[str, Any],
    *,
    backend_override: str | None = None,
    output_subdir: str = "generated",
) -> dict[str, Any]:
    """Generate every reference Han glyph through a pluggable backend.

    The backend supplies images; everything after that is the project's
    existing machinery. Each candidate image is normalized into the project's
    render geometry, swept across the configured thresholds to build a
    candidate family, scored by ``_evaluate_family`` against the reference
    structure and target style profile, and gated by the same hard topology
    validator the native path uses. A glyph whose learned candidates all fail
    falls back to the reference-derived candidate, so the run stays
    fail-complete exactly like ``generate_fusion_and_select``.
    """

    validate_data_flow_contract(cfg, require_prepared=True, write_report=True)
    work = Path(cfg["paths"]["work_dir"])
    analysis_path = work / "audit" / "analysis.csv"
    if not analysis_path.is_file():
        raise FileNotFoundError("missing audit/analysis.csv; run prepare first")
    rows = read_csv(analysis_path)
    if not rows:
        raise RuntimeError("analysis target list is empty")

    name = backend_name(cfg, backend_override)
    backend_cfg = cfg.get("backend", {})
    backend = create_backend(cfg, backend_override)
    backend.preflight()

    generated = ensure_dir(work / output_subdir)
    partial_path = generated / "selection.partial.csv"
    state_path = generated / "generation.state.json"
    selection_path = generated / "selection.csv"
    summary_path = generated / "summary.json"
    completion_path = generated / "generation.completed.json"
    fingerprint = _backend_fingerprint(cfg, analysis_path, name)

    if completion_path.is_file() and selection_path.is_file() and summary_path.is_file():
        try:
            completed = load_json(completion_path)
            existing = read_csv(selection_path)
            if completed.get("fingerprint") == fingerprint and len(existing) == len(rows):
                return load_json(summary_path)
        except Exception:
            pass

    selection_rows: list[dict[str, Any]] = []
    if partial_path.is_file() and state_path.is_file():
        try:
            state = load_json(state_path)
            if state.get("fingerprint") == fingerprint:
                selection_rows = read_csv(partial_path)
        except Exception:
            selection_rows = []
    if len({int(row["codepoint"]) for row in selection_rows}) != len(selection_rows):
        selection_rows = []

    chosen_dir = ensure_dir(generated / "chosen")
    candidate_root = ensure_dir(generated / "backend_candidates")
    style_pngs = _style_glyph_pngs(work, int(backend_cfg.get("zi2zi_jit", {}).get("style_pool_size", 64)))
    candidate_count = max(1, int(backend_cfg.get("candidate_count", 3)))

    request = BackendRequest(
        glyphs=tuple(
            GlyphRequest(
                int(row["codepoint"]),
                Path(row["ref_path"]),
                Path(row["ref_proxy_path"]),
            )
            for row in rows
        ),
        style_font=Path(cfg["paths"]["target_font"]),
        style_glyph_pngs=style_pngs,
        candidate_count=candidate_count,
        output_root=candidate_root,
        work_dir=work,
        reference_font=Path(cfg["paths"]["reference_font"]),
    )
    result = backend.generate(request)
    available = [read_candidate_dir(directory) for directory in result.candidate_dirs]
    if not available:
        raise BackendUnavailable(f"backend {name} produced no candidate directories")

    style_profile = load_json(work / "dataset" / "style_profile.json")
    profiles_path = work / "dataset" / "style_profiles.json"
    style_profiles = load_json(profiles_path) if profiles_path.is_file() else {"global": style_profile, "bins": []}
    thresholds_path = work / "audit" / "structure_thresholds.json"
    thresholds = load_json(thresholds_path) if thresholds_path.is_file() else {"keep": 0.05}
    # The backend gate is the global one with the backend overrides applied.
    # Structural invariants (component, hole and Euler delta) are inherited
    # unchanged; only the skeleton-similarity limits move.
    topology_cfg = deep_merge(cfg.get("topology", {}), backend_cfg.get("topology", {}))
    normal_inf = cfg.get("inference", {})
    fusion_inf = cfg.get("fusion", {}).get("inference", {})
    inference_size = int(fusion_inf.get("size", normal_inf.get("size", cfg["render"]["size"])))
    analysis_size = int(cfg["render"].get("analysis_size", 192))
    profile_size = int(cfg["render"].get("size", inference_size))
    maximum_border = float(normal_inf.get("maximum_border_ink", 0.015))
    threshold_offsets = [float(value) for value in fusion_inf.get(
        "threshold_offsets", normal_inf.get("threshold_offsets", [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10])
    )]
    # _confidence's constants are calibrated against the native gate. This path
    # is judged by the relaxed backend gate, so it must be told the ratio;
    # otherwise a candidate sitting comfortably inside the gate it was actually
    # accepted by still scores near zero. The reference fallback keeps the
    # native calibration because it is a reference-derived candidate and the
    # native scale is the right one for it.
    native_topology = cfg.get("topology", {})
    gate_relaxation = float(native_topology.get("maximum_topology_score", 0.06)) / max(
        float(topology_cfg.get("maximum_topology_score", 0.06)), 1e-6
    )
    delta_relaxation = float(native_topology.get("endpoint_tolerance_ratio", 0.16)) / max(
        float(topology_cfg.get("endpoint_tolerance_ratio", 0.16)), 1e-6
    )

    normalization = str(backend_cfg.get("normalization", "resample")).lower()
    # A backend that rasterized through its own renderer reports the transform
    # back into this project's frame. Honouring it is not optional: without it
    # every outline is placed wrongly inside the em box, so it overrides the
    # configured mode rather than being one more option.
    geometry: CandidateGeometry | None = None
    reported = result.metadata.get("geometry")
    if isinstance(reported, dict):
        geometry = CandidateGeometry.from_dict(reported)
        scale_to_inference = float(inference_size) / max(1.0, float(result.metadata.get("geometry_target_size", inference_size)))
        if abs(scale_to_inference - 1.0) > 1e-6:
            geometry = CandidateGeometry(
                scale=geometry.scale * scale_to_inference,
                offset_x=geometry.offset_x * scale_to_inference,
                offset_y=geometry.offset_y * scale_to_inference,
            )
        normalization = "affine"
    weights = {
        "backend": 1.00,
        "fallback": 1.12,
        **{str(key): float(value) for key, value in fusion_inf.get("candidate_weights", {}).items()},
    }
    progress_interval = max(1, int(fusion_inf.get("progress_checkpoint_interval", 32)))
    processed = {int(row["codepoint"]) for row in selection_rows}
    guard = LongRunGuard(cfg)
    progress = tqdm(
        total=len(rows), initial=len(selection_rows),
        desc=f"HanziStyleForge {name} selection", unit="glyph",
    )

    context = {
        "name": name,
        "inference_size": inference_size,
        "analysis_size": analysis_size,
        "profile_size": profile_size,
        "topology_cfg": topology_cfg,
        "threshold_offsets": threshold_offsets,
        "weights": weights,
        "maximum_border": maximum_border,
        "normalization": normalization,
        "geometry": geometry,
        "style_profiles": style_profiles,
        "keep_threshold": float(thresholds.get("keep", 0.05)),
        "gate_relaxation": gate_relaxation,
        "delta_relaxation": delta_relaxation,
        "render_threshold": float(cfg["render"]["threshold"]),
        "chosen_dir": str(chosen_dir),
    }
    tasks: list[tuple[dict[str, str], list[str]]] = []
    for row in rows:
        cp = int(row["codepoint"])
        if cp in processed:
            continue
        tasks.append((dict(row), [str(mapping[cp]) for mapping in available if cp in mapping]))

    workers = _resolve_worker_count(backend_cfg.get("selection_workers", 0))
    completed_since_checkpoint = 0
    try:
        # Selection is CPU-bound, per glyph independent, and on a full font it
        # dominates the wall clock: measured at roughly 59% of a 600-glyph run
        # against 41% for generation on the GPU. ProcessPoolExecutor.map keeps
        # results in submission order, so the resumable partial file is written
        # in exactly the same order a serial run would produce.
        if workers > 1 and len(tasks) > workers:
            executor = ProcessPoolExecutor(
                max_workers=workers, initializer=_init_selection_worker, initargs=(context,)
            )
            produced = executor.map(_select_glyph_task, tasks, chunksize=8)
        else:
            executor = None
            _init_selection_worker(context)
            produced = (_select_glyph_task(task) for task in tasks)
        try:
            for record in produced:
                selection_rows.append(record)
                processed.add(int(record["codepoint"]))
                progress.update(1)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= progress_interval:
                    _write_partial(partial_path, state_path, selection_rows, fingerprint, len(rows))
                    guard.checkpoint_boundary()
                    completed_since_checkpoint = 0
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
    finally:
        progress.close()
        if selection_rows and len(selection_rows) < len(rows):
            _write_partial(partial_path, state_path, selection_rows, fingerprint, len(rows))
            guard.checkpoint_boundary()

    selected_by_cp = {int(row["codepoint"]): row for row in selection_rows}
    missing = sorted({int(row["codepoint"]) for row in rows} - set(selected_by_cp))
    absent = [cp for cp, row in selected_by_cp.items() if not Path(str(row.get("chosen_path", ""))).is_file()]
    if missing or absent:
        save_json(generated / "coverage_failure.json", {"missing": missing, "absent_chosen_files": absent})
        raise RuntimeError(
            f"backend generation incomplete: missing={len(missing)}, absent_files={len(absent)}"
        )

    selection_rows = [selected_by_cp[int(row["codepoint"])] for row in rows]
    write_csv(selection_path, selection_rows, BACKEND_SELECTION_FIELDS)
    partial_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)

    sources: dict[str, int] = {}
    for item in selection_rows:
        key = str(item["chosen_source"])
        sources[key] = sources.get(key, 0) + 1
    save_codepoints(generated / "topology_failed.txt", [
        int(item["codepoint"]) for item in selection_rows if int(item["chosen_topology_pass"]) == 0
    ])
    save_codepoints(generated / "added.txt", [
        int(item["codepoint"]) for item in selection_rows if item["final_action"] == "add"
    ])
    summary = {
        "version": FUSION_CHECKPOINT_VERSION,
        "method": f"pluggable generation backend '{name}' + existing candidate selection and hard topology gate",
        "backend": result.metadata,
        "target_count": len(rows),
        "unique_output_count": len(selected_by_cp),
        "coverage_complete": True,
        "sources": sources,
        "topology_pass_count": sum(int(item["chosen_topology_pass"]) for item in selection_rows),
        "topology_failure_count": sum(1 - int(item["chosen_topology_pass"]) for item in selection_rows),
        "glyphs_without_backend_image": sum(
            1 for item in selection_rows if int(item.get("backend_candidate_count", 0) or 0) == 0
        ),
        "selection_csv": str(selection_path.resolve()),
        "fingerprint": fingerprint,
    }
    save_json(summary_path, summary)
    save_json(generated / "coverage.json", {
        "ref_han_target_count": len(rows),
        "selected_count": len(selection_rows),
        "unique_codepoint_count": len(selected_by_cp),
        "missing_count": 0,
        "chosen_file_missing_count": 0,
        "complete": True,
    })
    save_json(completion_path, {
        "version": FUSION_CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "target_count": len(rows),
    })
    return summary
