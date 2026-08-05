"""Report where a font file's bytes go, and how heavy its outlines are.

Contour simplification only helps if outlines are actually the bulk of the
file, so this prints the per-table byte breakdown first and the point-count
distribution second.  Run it against a hand-designed font to get a baseline
before deciding how far to push build.curve_simplify.

    python tools/font_size_report.py fonts/target.ttf
    python tools/font_size_report.py build/target-HanziStyleForge-Fusion.ttf
"""

from __future__ import annotations

import sys
from pathlib import Path

from fontTools.ttLib import TTFont


def _han(codepoints: set[int]) -> set[int]:
    blocks = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
              (0x20000, 0x2A6DF), (0x2A700, 0x2EBEF), (0x2F800, 0x2FA1F))
    return {c for c in codepoints if any(lo <= c <= hi for lo, hi in blocks)}


def main(path: str) -> None:
    target = Path(path)
    total = target.stat().st_size
    font = TTFont(target, lazy=False)
    print(f"{target.name}  {total / 1e6:.2f} MB  {font['maxp'].numGlyphs} glyphs")

    reader = font.reader
    rows = sorted(
        ((tag, reader.tables[tag].length) for tag in reader.keys() if tag in reader.tables),
        key=lambda row: row[1],
        reverse=True,
    )
    print("\nbytes by table")
    for tag, length in rows[:10]:
        print(f"  {tag:<6} {length / 1e6:>7.2f} MB  {100 * length / total:>5.1f}%")

    post = font["post"]
    print(f"\npost table format {post.formatType}"
          f"{'  <- stores a name per glyph; format 3.0 drops them' if post.formatType == 2.0 else ''}")

    if "glyf" not in font:
        print("\nCFF outlines: point statistics not comparable to TrueType.")
        return

    glyf = font["glyf"]
    reverse = font.getBestCmap()
    han = _han(set(reverse))
    names = {reverse[c] for c in han}

    counts: list[int] = []
    contours: list[int] = []
    for name in names:
        glyph = glyf[name]
        if glyph.numberOfContours <= 0:
            continue
        coordinates, end_points, _ = glyph.getCoordinates(glyf)
        counts.append(len(coordinates))
        contours.append(len(end_points))
    if not counts:
        print("\nno Han outlines found")
        return

    counts.sort()
    n = len(counts)
    pick = lambda q: counts[min(n - 1, int(q * n))]
    print(f"\nHan glyphs measured: {n}")
    print(f"  points per glyph   median {pick(0.5)}   p90 {pick(0.9)}   p99 {pick(0.99)}   max {counts[-1]}")
    print(f"  mean points/glyph  {sum(counts) / n:.1f}")
    print(f"  mean contours      {sum(contours) / len(contours):.1f}")
    print(f"  total Han points   {sum(counts) / 1e6:.2f} M")
    glyf_bytes = dict(rows).get("glyf", 0)
    if glyf_bytes:
        print(f"  bytes per point    {glyf_bytes / sum(counts):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fonts/target.ttf")
