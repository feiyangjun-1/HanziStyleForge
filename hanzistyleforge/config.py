from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import absolute_from, deep_merge, load_json, save_json


CHECKPOINT_FORMAT_VERSION = 301


DEFAULT_CONFIG: dict[str, Any] = {
    "version": CHECKPOINT_FORMAT_VERSION,
    "runtime": {
        "prevent_system_sleep": True,
        "durable_image_writes": False,
        "thermal_guard": {
            "enabled": True,
            "pause_above_c": 88,
            "resume_below_c": 80,
            "poll_seconds": 30,
        },
        "minimum_free_disk_gb": 35,
        "safe_stop_file": "STOP_AFTER_CHECKPOINT",
    },
    "paths": {
        "target_font": "fonts/target.ttf",
        "reference_font": "refs/ref.otf",
        "work_dir": "work_hanzistyleforge_fusion_months",
        "output_font": "build/target-HanziStyleForge-Fusion.ttf",
    },
    "scope": {
        "mode": "reference_han",
        "include_compatibility_ideographs": True,
        "extra_chars_file": "",
    },
    "render": {
        "size": 512,
        "analysis_size": 192,
        "proxy_skeleton_size": 240,
        "pad": 32,
        "antialias": 4,
        "threshold": 0.5,
        "canonical_radius_ratio": 0.0105,
        "distance_clip_ratio": 0.075,
    },
    "analysis": {
        "minimum_ink_ratio": 0.0025,
        "maximum_ink_ratio": 0.74,
        "maximum_border_ink": 0.015,
        "calibration_samples": 2048,
        "panel_count": 320,
        "maximum_style_glyphs": 0,
        "force_reprepare": False,
    },
    "retrieval": {
        "enabled": True,
        "grid": 5,
        "window_ratio": 0.34,
        "patches_per_glyph": 5,
        "maximum_patches": 120000,
        "stored_patch_size": 96,
        "descriptor_size": 10,
        "position_weight": 1.5,
        "minimum_activity": 0.008,
        "knn": 17,
        "flann_trees": 16,
        "flann_checks": 320,
        "distance_temperature": 0.9,
        "strength": 0.9,
    },
    "topology": {
        "analysis_size": 192,
        "prune_iterations": 1,
        "finalists_per_family": 6,
        "maximum_component_delta": 0,
        "maximum_hole_delta": 0,
        "maximum_euler_delta": 0,
        "minimum_endpoint_tolerance": 2,
        "minimum_junction_tolerance": 2,
        "endpoint_tolerance_ratio": 0.16,
        "junction_tolerance_ratio": 0.18,
        "maximum_missing_skeleton_p90": 0.023,
        "maximum_extra_skeleton_p90": 0.027,
        "maximum_hole_centroid_chamfer": 0.038,
        "maximum_zone_skeleton_distance": 0.155,
        "maximum_topology_score": 0.06,
        "structure_lock": True,
        "structure_lock_core_strength": 0.95,
        "structure_lock_radius_multiplier": 2.35,
    },
    "loss": {
        "weights": {
            "bce": 0.28,
            "dice": 0.24,
            "edge": 0.13,
            "multiscale": 0.08,
            "projection": 0.075,
            "cldice": 0.085,
            "proxy_skeleton": 0.003,
            "proxy_layout": 0.001,
            "sdf": 0.16,
            "skeleton_head": 0.10,
            "edge_head": 0.085,
            "topology_points": 0.003,
            "boundary_distance": 0.04,
            "style_signature": 0.12,
        }
    },
    "style_guard": {
        "enabled": True,
        "warmup_epochs": 16,
        "minimum_proxy_target_gap": 0.035,
        "minimum_style_direction": -0.03,
        "abort_on_collapse": True,
    },
    "adversarial": {
        "enabled": True,
        "base_channels": 24,
        "learning_rate": 5e-05,
        "weight": 0.0035,
        "feature_matching_weight": 0.30,
        "start_epoch": 50,
    },
    "training": {
        "seed": 20260718,
        "device": "cuda",
        "amp": True,
        "workers": 4,
        "cpu_threads": 6,
        "interop_threads": 1,
        "opencv_threads": 1,
        "image_cache_mb_per_process": 192,
        "prefetch_factor": 4,
        "validation_ratio": 0.06,
        "base_channels": 36,
        "weight_decay": 8e-05,
        "gradient_clip": 0.8,
        "ema_decay": 0.9995,
        "preview_every": 2,
        "checkpoint_every_steps": 400,
        "resume_if_exists": True,
        "reset_incompatible_checkpoints": True,
        "balanced_sampling": True,
        "samples_per_epoch": 0,
        "phases": [
            {
                "name": "foundation256",
                "size": 256,
                "epochs": 280,
                "batch_size": 6,
                "gradient_accumulation": 1,
                "learning_rate": 0.00018,
                "minimum_learning_rate": 1.5e-06,
                "early_stopping_patience": 64,
                "adversarial": False,
            },
            {
                "name": "structure384",
                "size": 384,
                "epochs": 220,
                "batch_size": 2,
                "gradient_accumulation": 3,
                "learning_rate": 4.5e-05,
                "minimum_learning_rate": 5e-07,
                "early_stopping_patience": 52,
                "adversarial": True,
                "adversarial_start_epoch": 50,
            },
            {
                "name": "finish512",
                "size": 512,
                "epochs": 160,
                "batch_size": 1,
                "gradient_accumulation": 8,
                "learning_rate": 1.2e-05,
                "minimum_learning_rate": 2.5e-07,
                "early_stopping_patience": 44,
                "adversarial": False,
            },
        ],
    },
    "refiner": {
        "enabled": True,
        "base_channels": 28,
        "size": 512,
        "epochs": 180,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "learning_rate": 2.5e-05,
        "early_stopping_patience": 48,
        "ema_decay": 0.9995,
        "samples_per_epoch": 0,
        "adversarial": False,
    },
    "inference": {
        "size": 512,
        "batch_size": 1,
        "threshold_offsets": [-0.13, -0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10, 0.13],
        "minimum_neural_confidence": 0.78,
        "minimum_pseudo_confidence": 0.99,
        "maximum_border_ink": 0.015,
        "generate_all_reference_glyphs": True,
        "allow_reference_fallback": True,
        "replacement_policy": "rebuild_all_reference",
        "allow_keep_target": False,
        "test_time_augmentation": True,
        "candidate_weights": {
            "neural": 1.0,
            "retrieval": 0.985,
            "fusion": 0.96,
            "fallback": 1.10,
        },
        "fusion_neural_weight": 0.64,
        "fusion_retrieval_weight": 0.36,
        "save_all_family_candidates": False,
            "progress_checkpoint_interval": 32,
        "resume_interval": 1,
    },
    "marathon": {
        "enabled": True,
        "cycles": 96,
        "hard_samples": 10000,
        "hard_validation_samples": 1024,
        "hard_eval_size": 256,
        "epochs_per_cycle": 18,
        "initial_learning_rate": 1.6e-05,
        "minimum_learning_rate": 5e-07,
        "learning_rate_decay": 0.965,
        "early_stop_cycles": 16,
        "minimum_dice_improvement": 0.00004,
        "snapshot_keep": 16,
        "ensemble": {
            "enabled": True,
            "independent_members": 3,
            "include_active_model": True,
            "include_marathon_snapshots": 3,
            "maximum_models": 7,
            "seed_stride": 104729,
            "use_for_inference": True,
        },
        "refine": {
            "enabled": True,
            "analysis_size": 192,
            "passes": 2,
            "global_search_trials": 96,
            "local_sweeps": 3,
            "zone_grid": 3,
            "save_every_glyphs": 16,
            "maximum_glyphs": 0,
            "use_component_layout": True,
            "decomposition_file": "data/cjkvi-ids/ids.txt",
            "auto_download": True,
            "source_url": "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/86b4d16159f0079437870408f0ca186e529015db/ids.txt",
            "source_sha256": "bfc70a8c09f9f5616ebf0543bd6681e67314e9f7ae2307e5ae8c6f15bdc5c6a6",
            "region_priority": [],
            "include_obsolete": False,
            "component_depth": 3,
            "maximum_component_zones": 24,
        },
    },
    "fusion": {
        "enabled": True,
        "style_dim": 256,
        "expert_count": 8,
        "latent_channels": 32,
        "vq_embeddings": 512,
        "diffusion_steps": 1000,
        "style_encoder": {
            "size": 128,
            "base_channels": 36,
            "heads": 8,
            # Both of these are needed to keep the expert set from collapsing
            # to a single repeated vector.  cell_grid gives the queries values
            # that actually differ; query_gain is the inverse temperature that
            # lets the softmax select among them.  Either one alone fails.
            "cell_grid": 4,
            "query_gain": 16.0,
            "references_per_set": 10,
            "inference_references": 16,
            "style_bank_groups": 12,
            "epochs": 220,
            "batch_size": 8,
            "virtual_length": 24000,
            "learning_rate": 0.00015,
            "weight_decay": 0.0001,
            "contrastive_margin": 0.35,
            "lr_patience": 10,
            "checkpoint_every_steps": 160,
            "early_stopping": {
                "enabled": True,
                "minimum_epochs": 100,
                "patience": 24,
                "minimum_relative_improvement": 0.002,
                "quality_window": 20,
                "positive_similarity_minimum": 0.995,
                # Separation is what the contrastive objective asks for and is
                # reachable; an absolute ceiling on the negative is not, because
                # positive and negative styles are drawn from the same
                # distribution.  The ceiling is kept only to catch an encoder
                # that has mapped everything together.
                "negative_similarity_maximum": 0.60,
                "separation_minimum": 0.35,
            },
        },
        "vq": {
            # channels_last measured 12% SLOWER for this autoencoder on an
            # RTX 5070 Ti Laptop, unlike the diffusion UNet where it helps, so
            # the two are configured separately rather than sharing a flag.
            "channels_last": False,
            # A code that stops being selected decays towards zero cluster size
            # while its running average decays identically, so the ratio that
            # defines it never moves and it can never be selected again. Any
            # code whose EMA count falls below this is reseeded onto a real
            # encoder output. 0 disables revival and restores the previous
            # behaviour, which collapsed a 512-entry codebook to a measured
            # perplexity near 7.
            "codebook_revive_threshold": 1.0,
            "base_channels": 52,
            "codebook_decay": 0.995,
            "commitment_weight": 0.24,
            "weight_decay": 0.00008,
            "preview_every": 4,
            "checkpoint_every_steps": 240,
            "validation_batches": 32,
            "loss_profile": "fast_balanced_v1",
            "fused_optimizer": False,
            "loss_weights": {
                "bce": 0.28, "dice": 0.26, "multiscale": 0.08,
                "projection": 0.06, "sdf": 0.15,
                "skeleton_head": 0.11, "edge_head": 0.12
            },
            "phases": [
                {"name": "vq256", "size": 256, "epochs": 220, "batch_size": 5, "gradient_accumulation": 1, "learning_rate": 0.00018, "minimum_learning_rate": 0.000001, "patience": 48},
                {"name": "vq384", "size": 384, "epochs": 150, "batch_size": 2, "gradient_accumulation": 3, "learning_rate": 0.000055, "minimum_learning_rate": 0.0000005, "patience": 42},
                {"name": "vq512", "size": 512, "epochs": 100, "batch_size": 1, "gradient_accumulation": 8, "learning_rate": 0.000016, "minimum_learning_rate": 0.00000025, "patience": 34}
            ]
        },
        "direct_baseline": {
            "enabled": True,
            "base_channels": 32,
            "ema_decay": 0.9995,
            "adversarial": False,
            "phases": [
                {"name": "direct256", "size": 256, "epochs": 160, "batch_size": 5, "gradient_accumulation": 1, "learning_rate": 0.00016, "minimum_learning_rate": 0.000001, "early_stopping_patience": 42, "adversarial": False},
                {"name": "direct384", "size": 384, "epochs": 100, "batch_size": 2, "gradient_accumulation": 3, "learning_rate": 0.00004, "minimum_learning_rate": 0.0000005, "early_stopping_patience": 34, "adversarial": False}
            ]
        },
        "diffusion": {
            "base_channels": 96,
            "content_base_channels": 40,
            "time_dim": 256,
            "schedule": "cosine",
            "style_dropout": 0.10,
            "ema_decay": 0.9999,
            "weight_decay": 0.00008,
            "noise_weight": 1.0,
            "latent_weight": 0.20,
            "image_weight": 0.30,
            "style_weight": 0.18,
            "style_loss_size": 128,
            "min_snr_gamma": 5.0,
            "gradient_clip": 1.0,
            "reconstruction_timestep_limit": 240,
            "validation_batches": 24,
            "preview_every": 4,
            "checkpoint_every_steps": 100,
            "image_loss_weights": {
                "bce": 0.24, "dice": 0.24, "edge": 0.14, "multiscale": 0.08,
                "projection": 0.06, "cldice": 0.07, "sdf": 0.16,
                "skeleton_head": 0.09, "edge_head": 0.08
            },
            # Batch sizes fold in the gradient accumulation these phases used
            # to carry: the effective batch is unchanged, but one optimizer
            # step replaces several and the GPU stops idling between them.
            # Measured on a 12 GB RTX 5070 Ti Laptop, latent256 went from
            # 0.0235 to 0.0114 s per sample at 1.5 GB of peak memory.
            "phases": [
                {"name": "latent256", "size": 256, "epochs": 280, "batch_size": 6, "gradient_accumulation": 1, "learning_rate": 0.00015, "minimum_learning_rate": 0.0000007, "patience": 64},
                {"name": "latent384", "size": 384, "epochs": 210, "batch_size": 6, "gradient_accumulation": 1, "learning_rate": 0.000045, "minimum_learning_rate": 0.00000035, "patience": 52},
                {"name": "latent512", "size": 512, "epochs": 150, "batch_size": 5, "gradient_accumulation": 2, "learning_rate": 0.000012, "minimum_learning_rate": 0.0000002, "patience": 44}
            ]
        },
        "purification": {
            "enabled": True,
            "cycles": 48,
            "hard_samples": 9000,
            "evaluation_size": 256,
            "size": 512,
            "epochs_per_cycle": 14,
            "batch_size": 1,
            "gradient_accumulation": 10,
            "hard_repeat": 7,
            "initial_learning_rate": 0.000009,
            "minimum_learning_rate": 0.00000018,
            "learning_rate_decay": 0.96,
            "epoch_patience": 10,
            "minimum_improvement": 0.00001,
            "minimum_cycle_improvement": 0.000004,
            "early_stop_cycles": 10
        },
        "refiner": {
            "enabled": True,
            "size": 512,
            "base_channels": 36,
            "epochs": 200,
            "batch_size": 1,
            "gradient_accumulation": 10,
            "learning_rate": 0.000022,
            "ema_decay": 0.9997,
            "style_weight": 0.20,
            "validation_batches": 24,
            "preview_every": 4,
            "checkpoint_every_steps": 100,
            "lr_patience": 10,
            "patience": 52,
            "loss_weights": {
                "bce": 0.25, "dice": 0.25, "edge": 0.16, "multiscale": 0.08,
                "projection": 0.07, "cldice": 0.08, "boundary_distance": 0.05
            }
        },
        "component_atlas": {
            "enabled": True,
            "decomposition_file": "data/cjkvi-ids/ids.txt",
            "auto_download": True,
            "source_url": "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/86b4d16159f0079437870408f0ca186e529015db/ids.txt",
            "source_sha256": "bfc70a8c09f9f5616ebf0543bd6681e67314e9f7ae2307e5ae8c6f15bdc5c6a6",
            "region_priority": [],
            "include_obsolete": False,
            "stored_patch_size": 96,
            "descriptor_size": 8,
            "maximum_patches": 220000,
            "maximum_per_component": 72,
            "maximum_depth": 3,
            "maximum_regions_per_glyph": 48,
            "minimum_activity": 0.005,
            "knn": 7,
            "strength": 0.94,
            "distance_temperature": 0.85
        },
        "contour_polisher": {
            "enabled": True,
            "points": 160,
            "hidden": 224,
            "layers": 8,
            "heads": 8,
            "maximum_training_contours": 180000,
            "virtual_length": 240000,
            "epochs": 220,
            "batch_size": 48,
            "learning_rate": 0.00012,
            "patience": 48,
            "build_strength": 0.56
        },
        "inference": {
            "size": 512,
            "diffusion_seeds": 3,
            "ddim_steps": 48,
            "seed_batch_size": 2,
            "ddim_eta": 0.0,
            "guidance_scale": 1.22,
            "snap_to_codebook": True,
            "threshold": 0.5,
            "threshold_offsets": [-0.12, -0.09, -0.06, -0.035, -0.018, 0.0, 0.018, 0.035, 0.06, 0.09, 0.12],
            "candidate_weights": {
                "diffusion": 0.90, "direct": 1.00, "retrieval": 0.98,
                "component": 0.94, "fusion": 0.88, "fallback": 1.12
            },
            "save_all_family_candidates": False
        }
    },
    "backend": {
        # "native" keeps the built-in Style Encoder -> VQ -> Diffusion -> Refiner
        # stack and is byte-for-byte the previous behaviour. Any other value
        # routes generation through hanzistyleforge.backends.
        "name": "native",
        "candidate_count": 3,
        # The long-run refiner optimizes towards the reference structure, so it
        # discards a style-transfer backend's glyphs. Off for non-native
        # backends; the native path is unaffected either way.
        "run_refine": False,
        # Candidate selection is CPU-bound and per glyph independent, and on a
        # full font it outweighs GPU generation: 59% against 41% of the wall
        # clock on a measured 600-glyph run. 0 picks half the cores, capped at
        # 8, since the stage saturates memory bandwidth before it saturates
        # cores.
        "selection_workers": 0,
        # "resample" is correct when the backend consumed content images that
        # this project rendered. "ref_bbox_fit" re-registers images of unknown
        # provenance, at the cost of the candidate's own proportions. A backend
        # that reports a measured geometry transform overrides this with
        # "affine", which is the only correct choice in that case.
        "normalization": "resample",
        # Topology gate overrides applied on top of the global topology block
        # for non-native backends only.
        #
        # The global thresholds are calibrated for the native generator, which
        # is structure-locked to the reference proxy and therefore reproduces
        # its skeleton closely. A backend performing genuine style transfer
        # deviates from that skeleton by design, and measured against a
        # style-transfer backend measured against it, the global gate rejected
        # topology_score ran at a median of 0.14 against a 0.06 limit and zone
        # skeleton distance at 0.26 against 0.155.
        #
        # The values below sit at roughly the 90th percentile of that measured
        # distribution. What is deliberately NOT relaxed is component, hole and
        # Euler delta: those stay at the global zero and are what actually
        # guarantees the generated glyph is the same character. In the same
        # measurement their medians were already zero, so honest style transfer
        # passes them while a glyph that gained or lost a stroke does not.
        "topology": {
            "maximum_topology_score": 0.30,
            "maximum_zone_skeleton_distance": 0.50,
            "maximum_missing_skeleton_p90": 0.090,
            "maximum_extra_skeleton_p90": 0.070,
            "maximum_hole_centroid_chamfer": 0.060,
            "minimum_endpoint_tolerance": 6,
            "minimum_junction_tolerance": 6,
            "endpoint_tolerance_ratio": 0.45,
            "junction_tolerance_ratio": 0.45,
        },
        "style_pool_size": 64,
        "dir": {
            "candidate_dirs": [],
            "require_complete": False,
        },
    },
    "benchmark": {
        "enabled": True,
        "maximum_glyphs": 1536,
        "batch_size": 1,
        "topology_sample_count": 1024,
        "minimum_reconstruction_dice": 0.93,
        "minimum_topology_pass_rate": 0.98,
        "warn_only": True,
    },
    "qa": {"contact_sheet_count": 960, "low_confidence_threshold": 0.75},
    "build": {
        "outline_mode": "sdf_quadratic",
        "sdf_upsample": 4,
        "sdf_sigma": 0.46,
        "sdf_levels": [0.0, -0.06, 0.06, -0.12, 0.12, -0.20, 0.20, -0.28, 0.28],
        "curve_simplify": 0.48,
        "corner_angle_degrees": 100.0,
        "maximum_points_per_contour": 480,
        "family_suffix": " HanziStyleForge Fusion",
        "postscript_suffix": "HanziStyleForgeFusion",
        "replace_strategy": "new_glyph_and_remap",
        "preserve_non_han": True,
        "verify_non_han_outlines": True,
        "remove_han_uvs": True,
        "disable_locl": False,
        "require_complete": True,
        "warn_on_bounds_anomaly": True,
    },
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    user = load_json(config_path) if config_path.exists() else {}
    cfg = deep_merge(DEFAULT_CONFIG, user)
    base = config_path.parent
    for key in ("target_font", "reference_font", "work_dir", "output_font"):
        cfg["paths"][key] = str(absolute_from(cfg["paths"][key], base))
    extra = cfg["scope"].get("extra_chars_file", "")
    if extra:
        cfg["scope"]["extra_chars_file"] = str(absolute_from(extra, base))
    stop_file = cfg.get("runtime", {}).get("safe_stop_file", "")
    if stop_file:
        cfg["runtime"]["safe_stop_file"] = str(absolute_from(stop_file, base))
    decomposition_file = cfg.get("marathon", {}).get("refine", {}).get("decomposition_file", "")
    if decomposition_file:
        cfg["marathon"]["refine"]["decomposition_file"] = str(absolute_from(decomposition_file, base))
    fusion_decomposition = cfg.get("fusion", {}).get("component_atlas", {}).get("decomposition_file", "")
    if fusion_decomposition:
        cfg["fusion"]["component_atlas"]["decomposition_file"] = str(absolute_from(fusion_decomposition, base))
    backend_cfg = cfg.get("backend", {})
    directories = backend_cfg.get("dir", {}).get("candidate_dirs", [])
    if directories:
        backend_cfg["dir"]["candidate_dirs"] = [
            str(absolute_from(item, base)) for item in directories if item
        ]
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(base)
    return cfg


def write_default_config(path: str | Path) -> None:
    save_json(path, DEFAULT_CONFIG)


def validate_config(cfg: dict[str, Any]) -> None:
    target = Path(cfg["paths"]["target_font"])
    reference = Path(cfg["paths"]["reference_font"])
    if not target.exists():
        raise FileNotFoundError(f"Target font not found: {target}\nCopy it to fonts/target.ttf or update the configuration.")
    if not reference.exists():
        raise FileNotFoundError(f"Reference font not found: {reference}\nCopy it to refs/ref.otf or update the configuration.")
    size = int(cfg["render"]["size"])
    pad = int(cfg["render"]["pad"])
    if size < 128 or size % 16 != 0 or pad < 4 or pad * 2 >= size:
        raise ValueError("render.size must be a multiple of 16 and at least 128; pad must be smaller than half the image size.")
    for phase in cfg["training"].get("phases", []):
        phase_size = int(phase["size"])
        if phase_size < 128 or phase_size % 16 != 0:
            raise ValueError("Each training-phase size must be a multiple of 16 and at least 128.")
    if not cfg["training"].get("phases"):
        raise ValueError("training.phases must not be empty.")
    if int(cfg["inference"].get("resume_interval", 1)) < 1:
        raise ValueError("inference.resume_interval must be at least 1.")
    if int(cfg["training"].get("checkpoint_every_steps", 150)) < 1:
        raise ValueError("training.checkpoint_every_steps must be at least 1.")
    if int(cfg["training"].get("workers", 0)) < 0:
        raise ValueError("training.workers must not be negative.")
    if str(cfg["inference"].get("replacement_policy", "")) != "rebuild_all_reference":
        raise ValueError("This release supports only rebuild_all_reference; existing Han outlines are not retained.")
    if bool(cfg["inference"].get("allow_keep_target", False)):
        raise ValueError("allow_keep_target=true is not permitted; every reference Han glyph is regenerated.")
    outline_mode = str(cfg["build"].get("outline_mode", "sdf_quadratic")).lower()
    if outline_mode not in {"sdf_quadratic", "quadratic_smooth", "polygon"}:
        raise ValueError("build.outline_mode is invalid.")
    replace_strategy = str(cfg["build"].get("replace_strategy", "adaptive_safe")).lower()
    if replace_strategy not in {"adaptive_safe", "new_glyph_and_remap", "overwrite_existing"}:
        raise ValueError("build.replace_strategy must be adaptive_safe, new_glyph_and_remap, or overwrite_existing.")
    ensemble = cfg.get("marathon", {}).get("ensemble", {})
    if int(ensemble.get("independent_members", 0)) < 0 or int(ensemble.get("maximum_models", 1)) < 1:
        raise ValueError("marathon.ensemble has an invalid member count.")
    backend_cfg = cfg.get("backend", {})
    backend_name = str(backend_cfg.get("name", "native")).lower()
    if backend_name not in {"native", "dir"}:
        raise ValueError(
            f"backend.name={backend_cfg.get('name')!r} is unknown. Available values: native, dir, zi2zi-jit."
        )
    if int(backend_cfg.get("candidate_count", 1)) < 1:
        raise ValueError("backend.candidate_count must be at least 1.")
    if str(backend_cfg.get("normalization", "resample")).lower() not in {"resample", "ref_bbox_fit"}:
        raise ValueError("backend.normalization must be resample or ref_bbox_fit.")
    fusion = cfg.get("fusion", {})
    if bool(fusion.get("enabled", True)):
        if int(fusion.get("latent_channels", 32)) < 8:
            raise ValueError("fusion.latent_channels must be at least 8.")
        if int(fusion.get("expert_count", 8)) < 2:
            raise ValueError("fusion.expert_count must be at least 2.")
        style_cfg = fusion.get("style_encoder", {})
        if int(style_cfg.get("batch_size", 1)) < 1:
            raise ValueError("fusion.style_encoder.batch_size must be at least 1.")
        if int(style_cfg.get("cell_grid", 4)) < 2:
            raise ValueError(
                "fusion.style_encoder.cell_grid must be at least 2; a 1x1 grid "
                "leaves the expert queries nothing to separate on."
            )
        if float(style_cfg.get("query_gain", 16.0)) < 8.0:
            raise ValueError(
                "fusion.style_encoder.query_gain must be at least 8; below that "
                "the expert attention softmax is flat and every expert reads "
                "out the same average, collapsing the expert set to one vector."
            )
        early_cfg = style_cfg.get("early_stopping", {})
        if bool(early_cfg.get("enabled", True)):
            if int(early_cfg.get("minimum_epochs", 1)) < 1:
                raise ValueError("Style early_stopping.minimum_epochs must be at least 1.")
            if int(early_cfg.get("patience", 1)) < 1:
                raise ValueError("Style early_stopping.patience must be at least 1.")
            if float(early_cfg.get("minimum_relative_improvement", 0.0)) < 0:
                raise ValueError("Style minimum_relative_improvement must not be negative.")
        for section in ("vq", "diffusion"):
            for phase in fusion.get(section, {}).get("phases", []):
                phase_size = int(phase.get("size", 0))
                if phase_size < 128 or phase_size % 16 != 0:
                    raise ValueError(f"fusion.{section}.phases size must be a multiple of 16 and at least 128.")
