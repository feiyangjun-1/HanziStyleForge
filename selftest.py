from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from hanzistyleforge.backends import (
    BackendRequest,
    BackendUnavailable,
    CandidateGeometry,
    DirectoryBackend,
    GlyphRequest,
    Zi2ziJitBackend,
    candidate_filename,
    codepoint_from_filename,
    normalize_candidate,
    read_candidate_dir,
)
from hanzistyleforge.build_font import _map_codepoint
from hanzistyleforge.features import expand_proxy_channels, make_target_aux, split_prediction
from hanzistyleforge.fusion_model import VectorQuantizerEMA
from hanzistyleforge.fusion_selftest import run_fusion_selftest
from hanzistyleforge.fusion_training import (
    _create_vq_optimizer,
    _prepare_vq_optimizer_state_dict,
    _repair_vq_optimizer_state_devices,
    _restore_vq_optimizer_backend,
    _style_plateau_state,
    _style_quality_gate,
)
from hanzistyleforge.contract import DataFlowContractError, validate_data_flow_contract
from hanzistyleforge.decomposition import (
    _token_codepoint,
    component_zones,
    load_decompositions,
    parse_ids_expression,
)
from hanzistyleforge.marathon_refine import run_marathon_refinement
from hanzistyleforge.losses import FontLossFinal, VQReconstructionLoss
from hanzistyleforge.backend_inference import _backend_fingerprint, _resolve_worker_count
from hanzistyleforge.inference import _confidence, _emergency_fallback_row
from hanzistyleforge.model import FontStyleNetFinal, GlyphRefinerFinal, PatchDiscriminatorFinal
from hanzistyleforge.proxy import (
    calibrate_observed_structure_thresholds,
    calibrate_same_structure_thresholds,
    ink_bbox,
    make_content_proxy,
    make_reference_fallbacks,
    proxy_structure_score,
    read_proxy,
    save_ink,
    save_proxy,
)
from hanzistyleforge.image_cache import read_gray_u8
from hanzistyleforge.report import _load_gray
from hanzistyleforge.retrieval import StyleAtlas, _descriptor, render_retrieval_candidate
from hanzistyleforge.runtime import configure_runtime
from hanzistyleforge.topology import topology_metrics, validate_topology
from hanzistyleforge.vectorize import image_to_ttglyph
from hanzistyleforge.util import deep_merge, save_json, write_csv
from hanzistyleforge.config import DEFAULT_CONFIG, load_config


def run_codebook_selftest() -> None:
    """A VQ code that stops winning must not be annihilated.

    The EMA update divides a decaying running average by a floored count, so an
    unselected code shrinks geometrically towards the origin and can never be
    chosen again. Measured on a real 1536-entry codebook trained for days:
    34 entries had ever been used, the other 1502 sat at exactly zero, with a
    median pairwise distance of zero between them.
    """

    def drifted(revive: float) -> tuple[float, int]:
        """Seed a codebook on a wide distribution, then train on a narrow one.

        Codes seeded away from the surviving mode stop being selected, which is
        the situation the EMA update destroys them in.
        """

        torch.manual_seed(0)
        quantizer = VectorQuantizerEMA(embeddings=64, dimension=8, decay=0.9, revive_threshold=revive)
        quantizer.train()
        torch.manual_seed(1)
        quantizer(torch.randn(256, 8).mul(20.0).view(256, 8, 1, 1))
        narrow = torch.full((256, 8), 4.0).view(256, 8, 1, 1) + torch.randn(256, 8, 1, 1) * 0.01
        for _ in range(300):
            quantizer(narrow)
        norms = quantizer.codebook.norm(dim=1)
        near = int(((quantizer.codebook - 4.0).norm(dim=1) < 1.0).sum())
        return float(norms.min()), near

    smallest, near = drifted(1.0)
    assert near == 64, f"revival should bring every code back onto the data, got {near}/64"
    assert smallest > 1.0, f"no code may be left at the origin, smallest norm {smallest:.2e}"

    # The old behaviour must remain reproducible, both so it can be restored and
    # so this test is demonstrating a real difference rather than asserting on
    # something that was never broken.
    smallest_off, near_off = drifted(0.0)
    assert near_off <= 2, f"without revival the codebook should collapse, {near_off}/64 survived"
    assert smallest_off < 1e-5, f"unselected codes should decay to zero, smallest {smallest_off:.2e}"

    # An already-collapsed checkpoint must still snap safely, because the whole
    # point of snapping is to move a latent onto a code some real glyph was
    # assigned to, and the origin is not one. Reproduces a measured checkpoint:
    # 34 live codes, the rest at the origin with a denormal cluster_size, which
    # is why the test is against epsilon rather than zero.
    collapsed = VectorQuantizerEMA(embeddings=64, dimension=8, decay=0.9)
    collapsed.eval()
    with torch.no_grad():
        collapsed.codebook.zero_()
        collapsed.cluster_size.fill_(1.401e-43)
        collapsed.codebook[:4] = torch.randn(4, 8) * 7.0
        collapsed.cluster_size[:4] = 100.0
    torch.manual_seed(0)
    latents = torch.randn(2048, 8).view(2048, 8, 1, 1)
    snapped, chosen = collapsed.nearest(latents)
    assert int(chosen.max()) < 4, "a dead code must never win the nearest-neighbour search"
    assert float(snapped.view(2048, 8).norm(dim=1).median()) > 1.0, (
        "snapping onto the origin collapses the decoder input to a constant field"
    )

    # A fresh codebook seeds itself from the data rather than the unit sphere.
    torch.manual_seed(0)
    seeded = VectorQuantizerEMA(embeddings=32, dimension=8, decay=0.9)
    seeded.train()
    far = torch.full((128, 8), 50.0).view(128, 8, 1, 1) + torch.randn(128, 8, 1, 1)
    assert float(seeded.codebook.norm(dim=1).median()) < 5.0
    seeded(far)
    assert float(seeded.codebook.norm(dim=1).median()) > 40.0, "first batch must reseed the codebook"

    # A resumed checkpoint keeps its learned codebook: a non-zero cluster_size
    # is what marks it as already initialized, so no new buffer is needed and
    # existing checkpoints still load strictly.
    resumed = VectorQuantizerEMA(embeddings=32, dimension=8, decay=0.9)
    resumed.load_state_dict(seeded.state_dict())
    resumed.train()
    before = resumed.codebook.clone()
    resumed(far)
    assert not torch.allclose(before, torch.zeros_like(before))
    assert set(seeded.state_dict()) == set(VectorQuantizerEMA(embeddings=32, dimension=8).state_dict())


def run_backend_selftest() -> None:
    """Exercise the pluggable generation-backend contract.

    The dir backend is deliberately model-free, so this covers the filename
    codec, candidate discovery and the 256 -> render.size normalization that
    every other backend will route through.
    """

    assert candidate_filename(0x4E00) == "U+4E00.png"
    assert candidate_filename(0x20000) == "U+20000.png"
    assert codepoint_from_filename("U+4E00.png") == 0x4E00
    # zi2zi-JiT's native output naming must be consumable without renaming.
    assert codepoint_from_filename("0000_U+4E00.png") == 0x4E00
    assert codepoint_from_filename("U+20B9F.png") == 0x20B9F
    assert codepoint_from_filename("metadata.json") is None

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference_dir = root / "ref"
        reference_dir.mkdir()
        first = root / "candidate_00"
        second = root / "candidate_01"
        first.mkdir()
        second.mkdir()

        # A cross that is offset and undersized inside its 256 frame, so the
        # two normalization modes provably differ.
        small = np.zeros((256, 256), dtype=np.float32)
        small[40:150, 80:96] = 1.0
        small[80:96, 40:150] = 1.0
        save_ink(first / "U+4E00.png", small)
        save_ink(first / "U+4E8C.png", small)
        # The second candidate covers only one of the two requested glyphs;
        # partial coverage must be reported, not raised.
        save_ink(second / "U+4E00.png", small)
        (second / "notes.txt").write_text("ignored", encoding="utf-8")

        centred = np.zeros((512, 512), dtype=np.float32)
        centred[100:412, 240:272] = 1.0
        centred[240:272, 100:412] = 1.0
        save_ink(reference_dir / "U+4E00.png", centred)

        discovered = read_candidate_dir(first)
        assert set(discovered) == {0x4E00, 0x4E8C}
        assert read_candidate_dir(second) == {0x4E00: second / "U+4E00.png"}

        duplicate = root / "duplicate"
        duplicate.mkdir()
        save_ink(duplicate / "U+4E00.png", small)
        save_ink(duplicate / "0007_U+4E00.png", small)
        try:
            read_candidate_dir(duplicate)
        except ValueError as exc:
            assert "U+4E00" in str(exc)
        else:
            raise AssertionError("a duplicated codepoint must be rejected")

        backend = DirectoryBackend([first, second])
        backend.preflight()
        request = BackendRequest(
            glyphs=(
                GlyphRequest(0x4E00, reference_dir / "U+4E00.png", reference_dir / "U+4E00.png"),
                GlyphRequest(0x4E8C, reference_dir / "U+4E00.png", reference_dir / "U+4E00.png"),
            ),
            style_font=Path("fonts/target.ttf"),
            style_glyph_pngs={},
            candidate_count=2,
            output_root=root / "out",
            work_dir=root,
        )
        result = backend.generate(request)
        assert len(result.candidate_dirs) == 2
        assert result.metadata["requested_glyphs"] == 2
        assert result.metadata["candidate_dirs"][0]["requested_covered"] == 2
        assert result.metadata["candidate_dirs"][1]["requested_covered"] == 1
        assert result.metadata["missing_from_all_candidates"] == 0

        resampled = normalize_candidate(first / "U+4E00.png", target_size=512, mode="resample")
        assert resampled.shape == (512, 512)
        assert 0.0 <= float(resampled.min()) and float(resampled.max()) <= 1.0
        fitted = normalize_candidate(
            first / "U+4E00.png",
            target_size=512,
            mode="ref_bbox_fit",
            reference_png=reference_dir / "U+4E00.png",
        )
        assert fitted.shape == (512, 512)
        reference_box = ink_bbox(np.asarray(read_gray_u8(reference_dir / "U+4E00.png")) < 128)
        fitted_box = ink_bbox(fitted >= 0.5)
        assert reference_box is not None and fitted_box is not None
        # ref_bbox_fit must land the ink on the reference box; plain resampling
        # keeps the original off-centre placement.
        assert all(abs(a - b) <= 3 for a, b in zip(fitted_box, reference_box))
        assert ink_bbox(resampled >= 0.5) != fitted_box

        try:
            DirectoryBackend([root / "does_not_exist"]).preflight()
        except BackendUnavailable as exc:
            assert "cannot find" in str(exc)
        else:
            raise AssertionError("a missing candidate directory must fail preflight")

        try:
            DirectoryBackend([]).preflight()
        except BackendUnavailable as exc:
            assert "candidate_dirs" in str(exc)
        else:
            raise AssertionError("an unconfigured dir backend must fail preflight")

        missing_request = BackendRequest(
            glyphs=(GlyphRequest(0x9F98, reference_dir / "U+4E00.png", reference_dir / "U+4E00.png"),),
            style_font=Path("fonts/target.ttf"),
            style_glyph_pngs={},
            candidate_count=1,
            output_root=root / "out",
            work_dir=root,
        )
        tolerant = DirectoryBackend([first]).generate(missing_request)
        assert tolerant.metadata["missing_from_all_candidates"] == 1
        try:
            DirectoryBackend([first], require_complete=True).generate(missing_request)
        except BackendUnavailable as exc:
            assert "U+9F98" in str(exc)
        else:
            raise AssertionError("require_complete must reject an uncovered glyph")

        _check_candidate_geometry(first, reference_dir)
        _check_backend_topology_override()
        _check_confidence_calibration()
        _check_selection_parallelism()
        _check_zi2zi_backend(root, reference_dir)


def _check_selection_parallelism() -> None:
    """Worker count must never influence the result or invalidate a resume."""

    assert _resolve_worker_count(4) == 4
    assert _resolve_worker_count(1) == 1
    auto = _resolve_worker_count(0)
    assert 1 <= auto <= 8, f"auto worker count {auto} outside the intended range"

    # selection_workers is a scheduling knob, so changing it must leave the
    # fingerprint alone; otherwise raising parallelism throws away a
    # part-finished multi-hour run.
    with tempfile.TemporaryDirectory() as temporary:
        analysis = Path(temporary) / "analysis.csv"
        analysis.write_text("codepoint\n19968\n", encoding="utf-8")
        base = {
            "backend": {"name": "dir", "candidate_count": 3, "selection_workers": 0},
            "topology": {"maximum_topology_score": 0.06},
            "render": {"size": 512},
            "inference": {},
        }
        other = {
            "backend": {"name": "dir", "candidate_count": 3, "selection_workers": 8},
            "topology": {"maximum_topology_score": 0.06},
            "render": {"size": 512},
            "inference": {},
        }
        assert _backend_fingerprint(base, analysis, "dir") == _backend_fingerprint(other, analysis, "dir")
        # A knob that does change the result still must.
        changed = {**other, "backend": {**other["backend"], "candidate_count": 5}}
        assert _backend_fingerprint(base, analysis, "dir") != _backend_fingerprint(changed, analysis, "dir")


def _check_confidence_calibration() -> None:
    """Confidence must be measured on the scale the candidate is judged by.

    Its constants are calibrated for the native gate. A candidate accepted by
    the relaxed backend gate scored near zero on that scale, so every backend
    glyph looked low-confidence and the QA report flagged an entire good run.
    """

    candidate = {
        "structure": {"structure_score": 0.16},
        "topology": {
            "topology_score": 0.14,
            "component_delta": 0,
            "hole_delta": 0,
            "endpoint_delta": 5,
            "junction_delta": 4,
        },
        "style_score": 0.20,
        "validation": {"hard_pass": True},
    }

    native = _confidence(candidate, None, 0.05)
    # Defaults must reproduce the native calibration exactly, so the built-in
    # generator's numbers cannot shift when a backend is configured.
    assert _confidence(candidate, None, 0.05, gate_relaxation=1.0, delta_relaxation=1.0) == native

    # 0.06/0.30 and 0.16/0.45 are the ratios between the native gate and the
    # shipped backend gate.
    relaxed = _confidence(candidate, None, 0.05, gate_relaxation=0.2, delta_relaxation=0.356)
    assert relaxed > native, "a relaxed gate must not still score like the strict one"
    assert relaxed > 10 * native, f"expected a large recalibration, got {native:.4f} -> {relaxed:.4f}"
    assert 0.0 <= relaxed <= 1.0

    # A candidate that is bad on the relaxed scale too must still score badly,
    # otherwise the recalibration is just inflating every number.
    bad = dict(candidate)
    bad["topology"] = dict(candidate["topology"])
    bad["topology"]["topology_score"] = 1.2
    bad["topology"]["endpoint_delta"] = 40
    bad["validation"] = {"hard_pass": False}
    assert _confidence(bad, None, 0.05, gate_relaxation=0.2, delta_relaxation=0.356) < 0.05

    profile = {"qa": {"low_confidence_threshold": 0.4}}
    assert float(profile["qa"]["low_confidence_threshold"]) == 0.4
    assert float(DEFAULT_CONFIG["qa"]["low_confidence_threshold"]) == 0.75


def _check_candidate_geometry(candidate_dir: Path, reference_dir: Path) -> None:
    """Cover the affine normalization used with a foreign rasterizer."""

    geometry = CandidateGeometry(scale=2.0, offset_x=-8.0, offset_y=12.0)
    assert CandidateGeometry.from_dict(geometry.as_dict()) == geometry

    source = candidate_dir / "U+4E00.png"
    fitted = normalize_candidate(
        source, target_size=512, mode="affine", geometry=geometry
    )
    assert fitted.shape == (512, 512)

    # The transform is measured from the backend's own resolution, so it must
    # be applied to the source pixels directly. Resizing to the target size
    # first and then scaling would double the magnification, which is a real
    # bug this asserts against: a 110px-wide mark at 256 scaled by 2.0 must
    # land at 220px, not 440px.
    box = ink_bbox(fitted >= 0.5)
    assert box is not None
    width = box[2] - box[0]
    assert 200 <= width <= 240, f"affine width {width} suggests a double-scaled candidate"

    try:
        normalize_candidate(source, target_size=512, mode="affine")
    except ValueError as exc:
        assert "geometry" in str(exc)
    else:
        raise AssertionError("affine mode without a geometry must be rejected")

    try:
        normalize_candidate(source, target_size=512, mode="nonsense")
    except ValueError as exc:
        assert "nonsense" in str(exc)
    else:
        raise AssertionError("an unknown normalization mode must be rejected")
    del reference_dir


def _check_backend_topology_override() -> None:
    """The backend gate must relax similarity limits but never the invariants.

    Component, hole and Euler delta are what guarantee the generated glyph is
    the same character. The looser skeleton limits exist because a style
    transfer backend deviates from the reference skeleton by design.
    """

    cfg = load_config("config.json") if Path("config.json").is_file() else {"topology": DEFAULT_CONFIG["topology"], "backend": DEFAULT_CONFIG["backend"]}
    merged = deep_merge(cfg["topology"], cfg["backend"]["topology"])
    assert merged["maximum_component_delta"] == cfg["topology"]["maximum_component_delta"] == 0
    assert merged["maximum_hole_delta"] == cfg["topology"]["maximum_hole_delta"] == 0
    assert merged["maximum_euler_delta"] == cfg["topology"]["maximum_euler_delta"] == 0
    assert merged["maximum_topology_score"] > cfg["topology"]["maximum_topology_score"]
    assert merged["maximum_zone_skeleton_distance"] > cfg["topology"]["maximum_zone_skeleton_distance"]
    assert merged["maximum_missing_skeleton_p90"] > cfg["topology"]["maximum_missing_skeleton_p90"]

    # A candidate that keeps the reference topology but shifts strokes must
    # pass the backend gate and fail the native one.
    reference = np.zeros((256, 256), dtype=np.float32)
    reference[40:216, 118:138] = 1.0
    reference[118:138, 40:216] = 1.0
    shifted = np.zeros((256, 256), dtype=np.float32)
    shifted[40:216, 128:148] = 1.0
    shifted[108:128, 40:216] = 1.0
    metrics = topology_metrics(reference, shifted, size=192, prune_iterations=1)
    assert int(metrics["component_delta"]) == 0
    assert validate_topology(metrics, merged)["hard_pass"]
    assert not validate_topology(metrics, cfg["topology"])["hard_pass"]


def _check_zi2zi_backend(root: Path, reference_dir: Path) -> None:
    """Cover the zi2zi-JiT backend paths that need no checkpoint.

    Generation itself requires a multi-gigabyte model, so the self-test
    exercises preflight diagnostics and the npz encoding instead: those are
    what silently corrupt a run, whereas a broken subprocess call fails loudly
    on the first chunk.
    """

    missing_repo = Zi2ziJitBackend(root / "no_repo", root / "no_checkpoint.pth")
    try:
        missing_repo.preflight()
    except BackendUnavailable as exc:
        assert "git clone" in str(exc)
    else:
        raise AssertionError("a missing zi2zi-JiT checkout must fail preflight")

    fake_repo = root / "fake_repo"
    fake_repo.mkdir()
    try:
        Zi2ziJitBackend(fake_repo, root / "no_checkpoint.pth").preflight()
    except BackendUnavailable as exc:
        assert "generate_chars.py is missing" in str(exc)
    else:
        raise AssertionError("a directory without generate_chars.py must fail preflight")

    (fake_repo / "generate_chars.py").write_text("", encoding="utf-8")
    try:
        Zi2ziJitBackend(fake_repo, root / "no_checkpoint.pth").preflight()
    except BackendUnavailable as exc:
        assert "Google Drive" in str(exc)
    else:
        raise AssertionError("a missing checkpoint must fail preflight")

    checkpoint = fake_repo / "weights.pth"
    checkpoint.write_bytes(b"not a checkpoint")
    try:
        Zi2ziJitBackend(
            fake_repo, checkpoint, python_executable=root / "no_python.exe"
        ).preflight()
    except BackendUnavailable as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("a missing interpreter must fail preflight")

    backend = Zi2ziJitBackend(fake_repo, checkpoint, font_label=7, chunk_size=2)
    # font_label is honoured without touching the checkpoint, so a LoRA
    # fine-tuned run never pays for the probe.
    assert backend.resolved_font_label() == 7

    style_png = root / "style.png"
    style = np.zeros((512, 512), dtype=np.float32)
    style[120:400, 240:270] = 1.0
    save_ink(style_png, style)

    request = BackendRequest(
        glyphs=(
            GlyphRequest(0x4E00, reference_dir / "U+4E00.png", reference_dir / "U+4E00.png"),
            GlyphRequest(0x20000, reference_dir / "U+4E00.png", reference_dir / "U+4E00.png"),
        ),
        style_glyph_pngs={0x4E8C: style_png, 0x4E09: style_png},
        style_font=Path("fonts/target.ttf"),
        candidate_count=1,
        output_root=root / "zi2zi_out",
        work_dir=root,
    )
    npz_path = root / "chunk.npz"
    backend._write_chunk_npz(npz_path, request.glyphs, [style_png], 1000, 0)
    with np.load(npz_path) as payload:
        assert payload["content_images"].shape == (2, 3, 256, 256)
        assert payload["content_images"].dtype == np.uint8
        assert payload["style_images"].shape == (2, 3, 128, 128)
        # generate_chars.py reads font_labels/unicode_labels to name outputs,
        # and LabelEmbedder has no character embedding at all.
        assert payload["font_labels"].tolist() == [1000, 1000]
        assert payload["char_labels"].tolist() == [0, 0]
        assert payload["unicode_labels"].tolist() == [0x4E00, 0x20000]

    empty_style = BackendRequest(
        glyphs=request.glyphs,
        style_glyph_pngs={},
        style_font=request.style_font,
        candidate_count=1,
        output_root=request.output_root,
        work_dir=root,
    )
    try:
        backend._style_pool(empty_style)
    except BackendUnavailable as exc:
        assert "prepare" in str(exc)
    else:
        raise AssertionError("an empty style pool must be rejected")


def main() -> None:
    runtime = configure_runtime({"training": {"cpu_threads": 1, "interop_threads": 1, "opencv_threads": 1}})
    assert runtime.get("opencv_threads") == 1
    assert runtime.get("torch_threads") == 1
    synthetic_history = [
        {
            "epoch": epoch,
            "val_loss": 0.03 if epoch < 5 else 0.02996,
            "val_positive_similarity": 0.9995,
            "val_negative_similarity": 0.11,
        }
        for epoch in range(1, 15)
    ]
    significant_best, last_significant, stale = _style_plateau_state(
        synthetic_history,
        through_epoch=14,
        minimum_relative_improvement=0.002,
    )
    assert significant_best == 0.03
    assert last_significant == 1 and stale == 13
    quality_ready, quality = _style_quality_gate(
        synthetic_history,
        window=10,
        minimum_positive_similarity=0.999,
        maximum_negative_similarity=0.15,
    )
    assert quality_ready
    assert quality["median_positive_similarity"] >= 0.999
    ink = np.zeros((96, 96), dtype=np.float32)
    ink[14:82, 43:53] = 1.0
    ink[43:53, 14:82] = 1.0
    # Add a closed counter so hole-position and Euler checks are exercised.
    ink[18:38, 18:38] = 1.0
    ink[23:33, 23:33] = 0.0

    proxy4 = make_content_proxy(ink, output_size=64, skeleton_size=64)
    proxy10 = expand_proxy_channels(proxy4)
    target_aux_np = make_target_aux(
        torch.nn.functional.interpolate(
            torch.from_numpy(ink[None, None]), size=(64, 64), mode="bilinear", align_corners=False
        )[0, 0].numpy()
    ).astype(np.float32) / 255.0
    assert proxy4.shape == (64, 64, 4)
    assert proxy10.shape == (64, 64, 10)
    assert target_aux_np.shape == (64, 64, 4)
    assert proxy_structure_score(proxy4, proxy4, analysis_size=64) < 1e-6
    thresholds = calibrate_same_structure_thresholds([proxy4] * 3, analysis_size=64)
    observed = calibrate_observed_structure_thresholds(
        [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.36, 0.38, 0.40] * 3,
        thresholds,
    )
    assert observed["very_strict"] < observed["uncertain"]

    topology_cfg = {
        "maximum_component_delta": 0,
        "maximum_hole_delta": 0,
        "maximum_euler_delta": 0,
        "minimum_endpoint_tolerance": 2,
        "minimum_junction_tolerance": 2,
        "maximum_missing_skeleton_p90": 0.05,
        "maximum_extra_skeleton_p90": 0.05,
        "maximum_hole_centroid_chamfer": 0.08,
        "maximum_zone_skeleton_distance": 0.32,
        "maximum_topology_score": 0.14,
    }
    topology = topology_metrics(ink, ink, size=64)
    assert validate_topology(topology, topology_cfg)["hard_pass"]
    fallbacks = dict(
        make_reference_fallbacks(
            ink,
            {
                "stroke_radius": {"median": 3.0},
                "bbox_width_ratio": {"median": 0.7},
                "bbox_height_ratio": {"median": 0.7},
                "center_x": {"median": 0.5},
                "center_y": {"median": 0.5},
            },
        )
    )
    assert "reference_raw" in fallbacks
    assert parse_ids_expression("⿱⿰日月木").serialize() == "⿱⿰日月木"
    assert _token_codepoint("⑦") == ord("⑦")
    assert _token_codepoint("19968") == 19968
    with tempfile.TemporaryDirectory() as ids_directory:
        ids_path = Path(ids_directory) / "ids.txt"
        ids_path.write_text(
            "# synthetic standard IDS test data\n"
            "U+660E\t明\t⿰日月\n"
            "U+4EAE\t亮\t⿱⿳亠口冖几[G]\t⿱亠兄[TJ]\n"
            "U+4E0D\t不\t⿱一③\n"
            "U+537F\t卿\t⿲𠂎⑦卩[K]\n",
            encoding="utf-8",
        )
        decompositions = load_decompositions(ids_path, region_priority=["G"])
        assert 0x660E in decompositions
        assert decompositions[0x4EAE].regions == ("G",)
        assert decompositions[0x4E0D].sequence == "⿱一③"
        assert decompositions[0x537F].sequence == "⿲𠂎⑦卩"
        zones = component_zones(0x660E, ink, decompositions, fallback_grid=3)
        assert len(zones) >= 2 and all(zone.shape == ink.shape for zone in zones)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        local_ids_path = root / "ids.txt"
        local_ids_path.write_text(
            "# synthetic standard IDS test data\nU+660E\t明\t⿰日月\n",
            encoding="utf-8",
        )
        proxy_path = root / "proxy.png"
        ink_path = root / "ink.png"
        atlas_path = root / "atlas.npz"
        save_proxy(proxy_path, proxy4)
        save_ink(ink_path, ink)
        loaded = read_proxy(proxy_path)
        assert loaded.shape == (64, 64, 4)
        assert _load_gray("", 32).size == (32, 32)
        emergency = _emergency_fallback_row(
            {
                "codepoint": str(0x4E00),
                "unicode": "U+4E00",
                "char": "一",
                "has_target": "0",
                "locl_sensitive": "0",
                "preliminary_status": "missing",
                "ref_path": str(ink_path),
                "target_path": "",
                "ref_proxy_path": str(proxy_path),
                "structure_score": "",
            },
            0x4E00,
            root,
            RuntimeError("self-test"),
        )
        assert emergency["chosen_label"] == "reference_emergency"
        assert Path(emergency["chosen_path"]).is_file()

        patch = loaded[8:56, 8:56]
        descriptor = _descriptor(patch, 0.5, 0.5, descriptor_size=6, position_weight=1.35)
        descriptors = np.stack(
            [descriptor + np.random.default_rng(index).normal(0, 0.01, descriptor.shape) for index in range(64)]
        ).astype(np.float32)
        mean = descriptors.mean(axis=0)
        std = np.maximum(descriptors.std(axis=0), 1e-4)
        normalized = (descriptors - mean) / std
        residuals = np.zeros((64, 32, 32), dtype=np.int8)
        residuals[:, 14:18, :] = 12
        np.savez_compressed(
            atlas_path,
            descriptors=normalized.astype(np.float16),
            descriptor_mean=mean,
            descriptor_std=std,
            residuals=residuals,
            source_codepoints=np.arange(0x4E00, 0x4E00 + 64, dtype=np.int32),
            source_xy=np.full((64, 2), 0.5, dtype=np.float16),
        )
        atlas = StyleAtlas.load(atlas_path, trees=2)
        # Final neural proxies have ten channels; retrieval deliberately slices to
        # the stable first four descriptor channels.
        retrieval, metadata = render_retrieval_candidate(
            proxy10,
            atlas,
            {
                "grid": 2,
                "window_ratio": 0.75,
                "descriptor_size": 6,
                "position_weight": 1.35,
                "minimum_activity": 0.001,
                "knn": 3,
                "flann_checks": 16,
                "strength": 0.8,
            },
        )
        assert retrieval.shape == (64, 64)
        assert metadata["query_count"] > 0

        work = root / "work"
        (work / "generated").mkdir(parents=True)
        (work / "audit").mkdir(parents=True)
        (work / "dataset").mkdir(parents=True)
        write_csv(
            work / "generated" / "selection.csv",
            [{
                "codepoint": 0x660E, "unicode": "U+660E", "char": "明",
                "chosen_source": "fallback", "final_action": "replace",
                "chosen_path": str(ink_path), "ref_path": str(ink_path),
                "ref_proxy_path": str(proxy_path), "notes": "",
            }],
            ["codepoint", "unicode", "char", "chosen_source", "final_action", "chosen_path", "ref_path", "ref_proxy_path", "notes"],
        )
        write_csv(
            work / "audit" / "analysis.csv",
            [{"codepoint": 0x660E, "complexity": 0.5}],
            ["codepoint", "complexity"],
        )
        save_json(work / "dataset" / "style_profile.json", {
            "stroke_radius": {"median": 3.0, "sigma": 1.0},
            "bbox_width_ratio": {"median": 0.7, "sigma": 0.1},
            "bbox_height_ratio": {"median": 0.7, "sigma": 0.1},
            "center_x": {"median": 0.5, "sigma": 0.05},
            "center_y": {"median": 0.5, "sigma": 0.05},
            "ink_ratio": {"median": 0.2, "sigma": 0.1},
        })
        save_json(work / "dataset" / "style_profiles.json", {"global": {
            "stroke_radius": {"median": 3.0, "sigma": 1.0},
            "bbox_width_ratio": {"median": 0.7, "sigma": 0.1},
            "bbox_height_ratio": {"median": 0.7, "sigma": 0.1},
            "center_x": {"median": 0.5, "sigma": 0.05},
            "center_y": {"median": 0.5, "sigma": 0.05},
            "ink_ratio": {"median": 0.2, "sigma": 0.1},
        }, "bins": []})
        refine_cfg = {
            "paths": {"work_dir": str(work)},
            "render": {"threshold": 0.5},
            "topology": topology_cfg | {"analysis_size": 64, "prune_iterations": 1},
            "marathon": {"refine": {
                "enabled": True, "analysis_size": 64, "passes": 1,
                "global_search_trials": 2, "local_sweeps": 1, "zone_grid": 2,
                "save_every_glyphs": 1, "maximum_glyphs": 0,
                "use_component_layout": True,
                "decomposition_file": str(local_ids_path),
                "auto_download": False,
            }},
        }
        refined_summary = run_marathon_refinement(refine_cfg)
        assert refined_summary["output_count"] == 1
        assert (work / "refined" / "selection.csv").is_file()

        glyph = image_to_ttglyph(
            ink_path,
            upm=1000,
            pad=8,
            y_bottom=-120,
            y_top=880,
            config={
                "outline_mode": "sdf_quadratic",
                "minimum_contour_area": 1.0,
                "sdf_upsample": 2,
                "sdf_sigma": 0.5,
                "sdf_levels": [0.0, -0.12, 0.12],
                "curve_simplify": 1.0,
                "corner_angle_degrees": 108.0,
                "maximum_points_per_contour": 128,
            },
        )
        assert int(glyph.numberOfContours) > 0

        # A supplementary-plane Han mapping must create a complete format-12
        # cmap.  It must not corrupt format-4 or hide existing BMP/non-Han text.
        cmap_font_path = root / "supplementary-cmap.ttf"
        cmap_saved_path = root / "supplementary-cmap-saved.ttf"
        builder = FontBuilder(1000, isTTF=True)
        glyph_order = [".notdef", "A", "han"]
        builder.setupGlyphOrder(glyph_order)
        simple_glyphs = {}
        for glyph_name in glyph_order:
            pen = TTGlyphPen(None)
            pen.moveTo((100, 100)); pen.lineTo((900, 100)); pen.lineTo((900, 900)); pen.lineTo((100, 900)); pen.closePath()
            simple_glyphs[glyph_name] = pen.glyph()
        builder.setupGlyf(simple_glyphs)
        builder.setupHorizontalMetrics({name: (1000, 0) for name in glyph_order})
        builder.setupHorizontalHeader(ascent=880, descent=-120)
        builder.setupCharacterMap({0x41: "A", 0x4E00: "han"})
        builder.setupOS2(sTypoAscender=880, sTypoDescender=-120, usWinAscent=900, usWinDescent=140)
        builder.setupNameTable({"familyName": "CmapSelftest", "styleName": "Regular"})
        builder.setupPost(); builder.setupMaxp(); builder.save(cmap_font_path)
        cmap_font = TTFont(cmap_font_path)
        _map_codepoint(cmap_font, 0x20000, "han")
        cmap_font.save(cmap_saved_path); cmap_font.close()
        cmap_verify = TTFont(cmap_saved_path)
        best_cmap = cmap_verify.getBestCmap() or {}
        assert {0x41, 0x4E00, 0x20000}.issubset(best_cmap)
        assert any(table.isUnicode() and table.format == 12 for table in cmap_verify["cmap"].tables)
        assert all(0x20000 not in (getattr(table, "cmap", {}) or {}) for table in cmap_verify["cmap"].tables if table.format == 4)
        cmap_verify.close()

        # Data-flow contract: training must use only target caches and final
        # generation must use only ref caches.  A ref path in the training CSV
        # must be rejected immediately.
        contract_work = root / "contract_work"
        target_proxy_dir = contract_work / "cache" / "target_proxy"
        target_render_dir = contract_work / "cache" / "target_render"
        target_aux_dir = contract_work / "cache" / "target_aux"
        ref_proxy_dir = contract_work / "cache" / "ref_proxy"
        ref_render_dir = contract_work / "cache" / "ref_render"
        for folder in (target_proxy_dir, target_render_dir, target_aux_dir, ref_proxy_dir, ref_render_dir):
            folder.mkdir(parents=True, exist_ok=True)
        for path in (
            target_proxy_dir / "U4E00.png", target_render_dir / "U4E00.png",
            target_aux_dir / "U4E00.png", ref_proxy_dir / "U4E00.png",
            ref_render_dir / "U4E00.png",
        ):
            path.write_bytes(b"selftest")
        (contract_work / "dataset").mkdir(parents=True, exist_ok=True)
        (contract_work / "audit").mkdir(parents=True, exist_ok=True)
        write_csv(
            contract_work / "dataset" / "index.csv",
            [{
                "sample_id": "style-self-U4E00", "codepoint": 0x4E00, "unicode": "U+4E00",
                "char": "一", "split": "train", "mode": "self",
                "proxy_path": str(target_proxy_dir / "U4E00.png"),
                "target_path": str(target_render_dir / "U4E00.png"),
                "target_aux_path": str(target_aux_dir / "U4E00.png"),
                "sample_weight": 1.0, "structure_score": 0.0, "complexity": 1.0,
            }],
            ["sample_id", "codepoint", "unicode", "char", "split", "mode", "proxy_path",
             "target_path", "target_aux_path", "sample_weight", "structure_score", "complexity"],
        )
        write_csv(
            contract_work / "audit" / "analysis.csv",
            [{
                "codepoint": 0x4E00, "unicode": "U+4E00", "char": "一",
                "ref_path": str(ref_render_dir / "U4E00.png"),
                "ref_proxy_path": str(ref_proxy_dir / "U4E00.png"),
            }],
            ["codepoint", "unicode", "char", "ref_path", "ref_proxy_path"],
        )
        contract_cfg = {"paths": {"work_dir": str(contract_work)}}
        contract_report = validate_data_flow_contract(contract_cfg, require_prepared=True, write_report=True)
        assert contract_report["passed"] and contract_report["cross_font_training_pairs"] == 0
        contaminated = list(__import__("csv").DictReader(
            (contract_work / "dataset" / "index.csv").open("r", encoding="utf-8-sig")
        ))
        contaminated[0]["proxy_path"] = str(ref_proxy_dir / "U4E00.png")
        write_csv(
            contract_work / "dataset" / "index.csv", contaminated,
            ["sample_id", "codepoint", "unicode", "char", "split", "mode", "proxy_path",
             "target_path", "target_aux_path", "sample_weight", "structure_score", "complexity"],
        )
        try:
            validate_data_flow_contract(contract_cfg, require_prepared=True, write_report=False)
        except DataFlowContractError:
            pass
        else:
            raise AssertionError("The data-flow contract accepted ref data in the training index.")

    generator = FontStyleNetFinal(base=2).eval()
    x = torch.from_numpy(np.moveaxis(proxy10, -1, 0))[None].float()
    target_aux = torch.from_numpy(np.moveaxis(target_aux_np, -1, 0))[None].float()
    target = target_aux[:, 0:1]
    with torch.no_grad():
        logits = generator(x)
    assert logits.shape == (1, 4, 64, 64)
    ink_logits, sdf_logits, skeleton_logits, edge_logits = split_prediction(logits)
    assert all(head is not None for head in (ink_logits, sdf_logits, skeleton_logits, edge_logits))
    loss, pieces = FontLossFinal()(logits, target, content_proxy=x, target_aux=target_aux)
    assert torch.isfinite(loss)
    for key in ("proxy_skeleton_loss", "sdf_loss", "skeleton_head_loss", "topology_point_loss", "style_signature_loss"):
        assert key in pieces and torch.isfinite(pieces[key])
    vq_loss, vq_pieces = VQReconstructionLoss()(logits, target, target_aux=target_aux)
    assert torch.isfinite(vq_loss)
    for key in ("sdf_loss", "skeleton_head_loss", "edge_head_loss"):
        assert key in vq_pieces and torch.isfinite(vq_pieces[key])

    # Regression: historical VQ checkpoints may contain foreach/fused backend
    # flags and moment tensors with a different memory format. Restore them into
    # the stable single-tensor AdamW path and verify that an update succeeds.
    legacy_model = torch.nn.Conv2d(3, 4, 3, padding=1).to(memory_format=torch.channels_last)
    legacy_optimizer = torch.optim.AdamW(legacy_model.parameters(), foreach=True)
    legacy_input = torch.randn(2, 3, 8, 8).to(memory_format=torch.channels_last)
    legacy_optimizer.zero_grad(set_to_none=True)
    legacy_model(legacy_input).square().mean().backward()
    legacy_optimizer.step()
    legacy_optimizer_state = legacy_optimizer.state_dict()
    for group in legacy_optimizer_state["param_groups"]:
        group["fused"] = True
        group["foreach"] = True
        group["capturable"] = True

    restored_model = torch.nn.Conv2d(3, 4, 3, padding=1).to(memory_format=torch.channels_last)
    restored_optimizer, fused_backend = _create_vq_optimizer(
        restored_model.parameters(),
        learning_rate=1e-3,
        weight_decay=1e-5,
        request_fused=True,
        device=torch.device("cpu"),
    )
    assert not fused_backend
    restored_optimizer.load_state_dict(
        _prepare_vq_optimizer_state_dict(legacy_optimizer_state, fused=fused_backend)
    )
    _restore_vq_optimizer_backend(restored_optimizer, fused=fused_backend)
    _repair_vq_optimizer_state_devices(restored_optimizer)
    assert all(group.get("fused") is False for group in restored_optimizer.param_groups)
    assert all(group.get("foreach") is False for group in restored_optimizer.param_groups)
    assert not getattr(restored_optimizer, "_step_supports_amp_scaling", False)
    restored_optimizer.zero_grad(set_to_none=True)
    restored_model(legacy_input).square().mean().backward()
    restored_optimizer.step()

    generator_ink = torch.sigmoid(ink_logits)
    refiner = GlyphRefinerFinal(base=2).eval()
    with torch.no_grad():
        refined = refiner(torch.cat([generator_ink, x], dim=1))
    assert refined.shape == (1, 4, 64, 64)
    assert torch.isfinite(refined).all()

    discriminator = PatchDiscriminatorFinal(base=4).eval()
    with torch.no_grad():
        score, features = discriminator(generator_ink, return_features=True)
    assert score.ndim == 4 and features
    run_fusion_selftest()
    run_codebook_selftest()
    run_backend_selftest()
    print("HanziStyleForge Fusion self-test: OK")


if __name__ == "__main__":
    main()
