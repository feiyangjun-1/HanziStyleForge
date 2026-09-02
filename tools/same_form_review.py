"""Build data/same_form_han.txt by reviewing target against ref, character by character.

The list decides which characters get a "reference structure to target glyph"
training pair, so a character whose two fonts draw different structures must
not be in it. No automatic check can make that call on its own: component,
hole and Euler counts are the only things a comparison of two differently
styled fonts can agree on, and they are identical for a glyph drawn in a
Japanese form, one with the wrong stroke terminal, and one with a truncated
stroke. Screening narrows the field; the eye decides.

Two steps.

    python tools/same_form_review.py render
        Renders every character both fonts cover as one side-by-side image,
        target on the left and ref on the right, into same_form_review/.
        Filenames carry the character, so a folder set to large icons is
        readable as-is.

    (delete the images whose two halves differ structurally)

    python tools/same_form_review.py collect
        Reads the filenames that survived and writes data/same_form_han.txt.

`render` also accepts --screen to pre-drop characters whose topology already
disagrees, which is worth it on a first pass over ten thousand glyphs. It is
a filter on the workload, not a verdict: everything it keeps still has to be
looked at.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hanzistyleforge.charset import is_han_ideograph
from hanzistyleforge.config import load_config, resolve_font_path
from hanzistyleforge.proxy import gray_to_ink
from hanzistyleforge.render import FontRenderer
from hanzistyleforge.topology import topology_metrics

REVIEW_DIR = ROOT / "same_form_review"
OUTPUT = ROOT / "data" / "same_form_han.txt"
NAME_PATTERN = re.compile(r"^U([0-9A-Fa-f]{4,6})_")

CELL = 220
GAP = 6
LABEL = 26


def _panel(ink: np.ndarray, size: int) -> np.ndarray:
    small = cv2.resize(ink.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    gray = ((1.0 - small.clip(0.0, 1.0)) * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def render(screen: bool) -> None:
    config = load_config(str(ROOT / "config.json"))
    render_config = config["render"]
    target = FontRenderer(
        str(resolve_font_path(ROOT / config["paths"]["target_font"])),
        size=render_config["size"], pad=render_config["pad"], antialias=render_config["antialias"],
    )
    reference = FontRenderer(
        str(resolve_font_path(ROOT / config["paths"]["reference_font"])),
        size=render_config["size"], pad=render_config["pad"], antialias=render_config["antialias"],
    )

    shared = sorted(
        cp for cp in set(target.cmap) & set(reference.cmap) if is_han_ideograph(cp, True)
    )
    print(f"{len(shared)} characters are covered by both fonts")

    REVIEW_DIR.mkdir(exist_ok=True)
    width = CELL * 2 + GAP
    height = CELL + LABEL
    written = blank = screened = 0

    for index, cp in enumerate(shared, 1):
        target_ink = gray_to_ink(target.render_gray(cp))
        reference_ink = gray_to_ink(reference.render_gray(cp))
        if float(target_ink.sum()) == 0 or float(reference_ink.sum()) == 0:
            blank += 1
            continue

        if screen:
            metrics = topology_metrics(reference_ink, target_ink, size=192, prune_iterations=1)
            if any(int(metrics[key]) for key in ("component_delta", "hole_delta", "euler_delta")):
                screened += 1
                continue

        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[LABEL:, 0:CELL] = _panel(target_ink, CELL)
        canvas[LABEL:, CELL + GAP:] = _panel(reference_ink, CELL)
        canvas[LABEL:, CELL:CELL + GAP] = (200, 200, 200)
        cv2.putText(canvas, "target", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ref", (CELL + GAP + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        path = REVIEW_DIR / ("U%04X_%s.png" % (cp, chr(cp)))
        path.write_bytes(cv2.imencode(".png", canvas)[1].tobytes())
        written += 1
        if index % 2000 == 0:
            print(f"  ...{index}/{len(shared)}")

    print(f"\nwrote {written} images to {REVIEW_DIR}")
    if blank:
        print(f"  {blank} skipped: blank in one of the fonts")
    if screened:
        print(f"  {screened} dropped by --screen: topology already disagrees")
    print("\nNow delete the images whose two halves differ structurally, then run:")
    print("  python tools/same_form_review.py collect")


def collect() -> None:
    if not REVIEW_DIR.is_dir():
        raise SystemExit(f"{REVIEW_DIR} does not exist. Run the render step first.")

    codepoints: list[int] = []
    unparsed: list[str] = []
    for path in sorted(REVIEW_DIR.glob("*.png")):
        match = NAME_PATTERN.match(path.name)
        if match:
            codepoints.append(int(match.group(1), 16))
        else:
            unparsed.append(path.name)

    if not codepoints:
        raise SystemExit(f"No parsable images left in {REVIEW_DIR}. Nothing written.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(chr(cp) for cp in codepoints), encoding="utf-8")

    print(f"{len(codepoints)} images remained in {REVIEW_DIR}")
    if unparsed:
        print(f"  ignored {len(unparsed)} files not named U<hex>_...: {unparsed[:5]}")
    print(f"wrote {len(codepoints)} characters to {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    render_command = commands.add_parser("render", help="render side-by-side comparisons for review")
    render_command.add_argument(
        "--screen",
        action="store_true",
        help="skip characters whose component, hole or Euler counts already disagree",
    )
    commands.add_parser("collect", help="write data/same_form_han.txt from the images that remain")

    args = parser.parse_args()
    if args.command == "render":
        render(args.screen)
    else:
        collect()


if __name__ == "__main__":
    main()
