"""Vectorize a directory of generated glyph PNGs into SVG files.

Reuses the project's own outline extraction -- multi-level SDF thresholding,
Douglas-Peucker simplification and corner-aware quadratic fitting -- and swaps
only the pen, so the curves match what build_font would put in the TTF.

Usage:
    .venv/Scripts/python.exe png_to_svg.py <input_dir> [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen

sys.path.insert(0, str(Path(__file__).parent))

from hanzistyleforge.config import load_config
from hanzistyleforge.vectorize import (
    _choose_sdf_mask,
    _contour_depth,
    _deduplicate,
    _limit_contour,
    _linear_path,
    _quadratic_path_corner_aware,
    _signed_area,
)

UPM = 1000


def read_image_unicode_safe(path: Path) -> np.ndarray | None:
    """cv2.imread cannot open non-ASCII paths on Windows."""
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)


def image_to_svg_path(image: np.ndarray, config: dict) -> str:
    if image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    ink = 1.0 - image.astype(np.float32) / 255.0
    if float(ink.mean()) > 0.5:  # white glyph on black, invert
        ink = 1.0 - ink

    mode = str(config.get("outline_mode", "sdf_quadratic")).lower()
    minimum_area = float(config.get("minimum_contour_area", 2.0))
    curve_simplify = float(config.get("curve_simplify", 1.05))
    maximum_points = int(config.get("maximum_points_per_contour", 320))
    corner_angle = float(config.get("corner_angle_degrees", 112.0))

    if mode == "sdf_quadratic":
        factor = max(1, int(config.get("sdf_upsample", 4)))
        sigma = max(0.0, float(config.get("sdf_sigma", 0.70))) * factor
        levels = [float(v) for v in config.get("sdf_levels", [0.0, -0.18, 0.18])]
        contour_mask = _choose_sdf_mask(ink, factor, sigma, levels)
        epsilon = curve_simplify * factor
        area_scale = factor * factor
    else:
        factor = 1
        contour_mask = (ink >= 0.5).astype(np.uint8) * 255
        epsilon = float(config.get("simplify", 0.85))
        area_scale = 1

    contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pen = SVGPathPen({}, ntos=lambda v: f"{v:.2f}")
    if hierarchy is None:
        return ""

    height, width = ink.shape
    for index, contour in enumerate(contours):
        if abs(float(cv2.contourArea(contour))) < minimum_area * area_scale:
            continue
        approx = _limit_contour(contour, epsilon=epsilon, maximum_points=maximum_points)
        if len(approx) < 3:
            continue
        pixels = np.asarray(approx[:, 0, :], dtype=np.float32) / float(factor)
        # Map the whole image onto the em square. These images come from another
        # pipeline and carry their own margins, so the project's render.pad
        # convention does not apply.
        raw = [(float(px) / width * UPM, UPM - float(py) / height * UPM) for px, py in pixels]
        points = _deduplicate(raw)
        if len(points) < 3:
            continue
        is_hole = _contour_depth(index, hierarchy) % 2 == 1
        if (_signed_area(points) > 0) != is_hole:
            points.reverse()
        if mode in {"sdf_quadratic", "quadratic_smooth"}:
            _quadratic_path_corner_aware(pen, points, corner_angle=corner_angle)
        else:
            _linear_path(pen, points)
    return pen.getCommands()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else source.parent / (source.name.replace("png", "svg") + ("" if "png" in source.name else "_svg"))
    destination.mkdir(parents=True, exist_ok=True)

    config = dict(load_config("config.json")["build"])
    # curve_simplify is a Douglas-Peucker tolerance in source pixels, and the
    # project's value is tuned for its own 512 px renders. On a smaller image
    # the same absolute tolerance is proportionally looser, and the quadratic
    # fitting then bulges between the retained points: measured on a 256 px
    # input, a straight stroke edge deviated 0.267 px RMS at 1.2 against the
    # source's 0.000, and 0.544 at 2.5. Scaling the tolerance by the image size
    # brings it to 0.084. Blurring slightly harder recovers some of the point
    # count that costs.
    probe = read_image_unicode_safe(files_probe) if (files_probe := next(iter(sorted(source.glob("*.png"))), None)) else None
    if probe is not None:
        scale = probe.shape[0] / 512.0
        config["curve_simplify"] = float(config.get("curve_simplify", 1.05)) * scale
        config["sdf_sigma"] = float(config.get("sdf_sigma", 0.70)) * max(1.0, 1.0 / max(scale, 1e-6)) ** 0.5
        print(f"input is {probe.shape[0]} px: curve_simplify -> {config['curve_simplify']:.2f}, "
              f"sdf_sigma -> {config['sdf_sigma']:.2f}")
    files = sorted(p for p in source.glob("*.png"))
    print(f"vectorizing {len(files)} images -> {destination}")

    written = skipped = 0
    for i, path in enumerate(files, 1):
        image = read_image_unicode_safe(path)
        if image is None:
            skipped += 1
            continue
        commands = image_to_svg_path(image, config)
        if not commands.strip():
            skipped += 1
            continue
        char = path.stem
        title = f"U+{ord(char):04X}" if len(char) == 1 else char
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {UPM} {UPM}" width="{UPM}" height="{UPM}">\n'
            f'  <title>{title}</title>\n'
            f'  <g transform="translate(0,{UPM}) scale(1,-1)">\n'
            f'    <path d="{commands}" fill="black" fill-rule="nonzero"/>\n'
            f'  </g>\n'
            f'</svg>\n'
        )
        (destination / f"{char}.svg").write_text(svg, encoding="utf-8")
        written += 1
        if i % 400 == 0:
            print(f"  ...{i}/{len(files)}")

    print(f"done: {written} written, {skipped} skipped")


if __name__ == "__main__":
    main()
