"""Hand-curate data/same_form_han.txt: do target and ref draw the same glyph form?

The list decides which characters get a "reference structure to target glyph"
training pair, so a character whose two fonts draw different structures must
not be in it. No automatic check can make that call: component, hole and Euler
counts are the only things two differently styled fonts can be compared on, and
they are identical for a glyph drawn in a Japanese form, one with the wrong
stroke terminal, and one with a truncated stroke. The eye is the judge.

Shows every Han character both fonts render, side by side, and records a verdict
per character. Every keystroke rewrites both the progress file and the output
list through a temporary file, so closing the window, killing the process or
losing power costs at most the keystroke in flight. Reopening resumes at the
first undecided character.

Keys
----
  y / 1 / space  same form -> goes in the list
  n / 0 / x      different form
  left / right   move without deciding
  u              clear the verdict on the current character
  A              mark every remaining undecided character as same (asks first)
  ctrl+s         report where the list was written (saving is automatic anyway)

Usage
-----
    python tools/same_form_review.py
    python tools/same_form_review.py --only-undecided
    python tools/same_form_review.py --sort suspicious

A topology score is shown for information. Do not lean on it: it cannot see the
form differences that matter, which is the reason this tool exists. Characters
are presented in codepoint order by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hanzistyleforge.charset import is_han_ideograph
from hanzistyleforge.config import load_config, resolve_font_path
from hanzistyleforge.proxy import gray_to_ink
from hanzistyleforge.render import FontRenderer
from hanzistyleforge.topology import topology_metrics

STATE_PATH = ROOT / "same_form_review_state.json"
OUTPUT = ROOT / "data" / "same_form_han.txt"
SCORE_CACHE = ROOT / "same_form_review_scores.json"

SAME, DIFF = "same", "diff"
SCORE_SIZE = 192

BG = "#1e1e1e"
FG = "#e8e8e8"
DIM = "#888888"
OK = "#4ec9b0"
NO = "#f48771"


def _ink_to_image(ink: np.ndarray, size: int) -> Image.Image:
    gray = ((1.0 - np.asarray(ink, dtype=np.float32).clip(0.0, 1.0)) * 255.0).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB").resize((size, size), Image.LANCZOS)


def compute_scores(codepoints, target, reference, key):
    """Topology disagreement per character, cached. A hint for ordering only."""
    if SCORE_CACHE.is_file():
        try:
            blob = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
            if blob.get("key") == key:
                return {int(k): v for k, v in blob["scores"].items()}
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    print(f"scoring {len(codepoints)} pairs (one-off, cached afterwards)...")
    scores = {}
    for index, cp in enumerate(codepoints, 1):
        target_ink = gray_to_ink(target.render_gray(cp))
        reference_ink = gray_to_ink(reference.render_gray(cp))
        if float(target_ink.sum()) == 0 or float(reference_ink.sum()) == 0:
            scores[cp] = 1.0
        else:
            metrics = topology_metrics(reference_ink, target_ink, size=SCORE_SIZE, prune_iterations=1)
            scores[cp] = float(metrics["topology_score"])
        if index % 1000 == 0:
            print(f"  {index}/{len(codepoints)}")

    SCORE_CACHE.write_text(
        json.dumps({"key": key, "scores": {str(k): v for k, v in scores.items()}}),
        encoding="utf-8",
    )
    return scores


class Reviewer:
    def __init__(self, root, items, target, reference, only_undecided, panel):
        self.root = root
        self.panel = panel
        self.target = target
        self.reference = reference
        self.state = self._load_state()

        if only_undecided:
            items = [item for item in items if chr(item[0]) not in self.state]
            if not items:
                messagebox.showinfo("done", "Nothing left to review.")
                root.destroy()
                return
        self.items = items
        self.index = self._first_undecided()
        self._photos = []

        root.title("same glyph form?   target  vs  ref")
        root.configure(bg=BG)

        self.header = tk.Label(root, bg=BG, fg=FG, font=("Consolas", 13), pady=6)
        self.header.pack(fill="x")
        self.canvas = tk.Label(root, bg=BG)
        self.canvas.pack(padx=10)
        self.verdict = tk.Label(root, bg=BG, fg=DIM, font=("Consolas", 12), pady=4)
        self.verdict.pack(fill="x")
        tk.Label(
            root, bg=BG, fg=DIM, font=("Consolas", 10), pady=6,
            text="y/space same   n/x different   left/right move   u undo   "
                 "A accept rest   ctrl+s where",
        ).pack(fill="x")

        for key in ("y", "Y", "1"):
            root.bind(key, lambda event: self._decide(SAME))
        root.bind("<space>", lambda event: self._decide(SAME))
        for key in ("n", "N", "0", "x", "X"):
            root.bind(key, lambda event: self._decide(DIFF))
        root.bind("<Left>", lambda event: self._move(-1))
        root.bind("<Right>", lambda event: self._move(1))
        root.bind("u", lambda event: self._undo())
        root.bind("A", lambda event: self._accept_rest())
        root.bind("<Control-s>", lambda event: self._export(verbose=True))
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._render()

    def _load_state(self):
        if STATE_PATH.is_file():
            try:
                return json.loads(STATE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        # No progress file, but a list may already exist from an earlier review
        # or from another machine. Seed from it rather than starting empty:
        # every export rewrites the list, so an empty start would erase it on
        # the first keystroke or on close.
        if OUTPUT.is_file():
            try:
                existing = [ch for ch in OUTPUT.read_text(encoding="utf-8") if ch.strip()]
            except OSError:
                existing = []
            if existing:
                backup = OUTPUT.with_suffix(".txt.bak")
                backup.write_text("".join(existing), encoding="utf-8")
                print(f"seeded {len(existing)} characters from {OUTPUT.name} "
                      f"(backed up to {backup.name})")
                return {ch: SAME for ch in existing}
        return {}

    def _save_state(self):
        """Rewrite progress and the list after every decision.

        Both go through a temporary file and an atomic replace, so a crash
        mid-write cannot leave either one truncated.
        """
        temporary = STATE_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=0), encoding="utf-8")
        os.replace(temporary, STATE_PATH)
        self._export()

    def _first_undecided(self):
        for i, (cp, _) in enumerate(self.items):
            if chr(cp) not in self.state:
                return i
        return 0

    def _render(self):
        cp, score = self.items[self.index]
        ch = chr(cp)
        size = self.panel
        gap = 10

        strip = Image.new("RGB", (size * 2 + gap, size), (30, 30, 30))
        panels = [
            _ink_to_image(gray_to_ink(self.target.render_gray(cp)), size),
            _ink_to_image(gray_to_ink(self.reference.render_gray(cp)), size),
        ]
        for i, image in enumerate(panels):
            strip.paste(image, (i * (size + gap), 0))
        draw = ImageDraw.Draw(strip)
        draw.text((4, 4), "target", fill=(120, 120, 120))
        draw.text((size + gap + 4, 4), "ref", fill=(120, 120, 120))

        photo = ImageTk.PhotoImage(strip)
        self._photos = [photo]
        self.canvas.configure(image=photo)

        same_count = sum(1 for v in self.state.values() if v == SAME)
        self.header.configure(
            text=f"  {ch}   U+{cp:04X}   topo {score:.3f}        "
                 f"{self.index + 1} / {len(self.items)}        "
                 f"decided {len(self.state)}   same {same_count}"
        )
        verdict = self.state.get(ch)
        self.verdict.configure(
            text={SAME: ">> SAME FORM <<", DIFF: ">> DIFFERENT <<"}.get(verdict, "undecided"),
            fg={SAME: OK, DIFF: NO}.get(verdict, DIM),
        )

    def _decide(self, value):
        self.state[chr(self.items[self.index][0])] = value
        self._save_state()
        self._move(1)

    def _undo(self):
        if self.state.pop(chr(self.items[self.index][0]), None) is not None:
            self._save_state()
        self._render()

    def _accept_rest(self):
        pending = [chr(cp) for cp, _ in self.items[self.index:] if chr(cp) not in self.state]
        if not pending:
            messagebox.showinfo("nothing to do", "No undecided characters from here on.")
            return
        if not messagebox.askyesno(
            "accept the rest?",
            f"Mark {len(pending)} undecided characters from position {self.index + 1} "
            f"onward as SAME FORM?\n\nUndo one at a time with 'u'.",
        ):
            return
        for ch in pending:
            self.state[ch] = SAME
        self._save_state()
        self._render()

    def _move(self, step):
        self.index = max(0, min(len(self.items) - 1, self.index + step))
        self._render()

    def _export(self, verbose=False):
        same = sorted(ch for ch, v in self.state.items() if v == SAME)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(".txt.tmp")
        temporary.write_text("".join(same), encoding="utf-8")
        os.replace(temporary, OUTPUT)
        if verbose:
            messagebox.showinfo("saved", f"{len(same)} same-form characters -> {OUTPUT}")
        return len(same)

    def _on_close(self):
        count = self._export()
        print(f"decided {len(self.state)} / {len(self.items)}, {count} marked same")
        print(f"output -> {OUTPUT}")
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--only-undecided", action="store_true")
    parser.add_argument(
        "--sort", default="codepoint", choices=["codepoint", "suspicious"],
        help="codepoint order by default. 'suspicious' puts the highest topology "
             "disagreement first, which finds obvious mismatches quickly but "
             "cannot rank the differences this tool exists to catch.",
    )
    parser.add_argument("--panel", type=int, default=300, help="Panel size in pixels.")
    args = parser.parse_args()

    config = load_config(args.config)
    render_config = config["render"]
    target_path = resolve_font_path(ROOT / config["paths"]["target_font"])
    reference_path = resolve_font_path(ROOT / config["paths"]["reference_font"])
    target = FontRenderer(
        str(target_path), size=render_config["size"],
        pad=render_config["pad"], antialias=render_config["antialias"],
    )
    reference = FontRenderer(
        str(reference_path), size=render_config["size"],
        pad=render_config["pad"], antialias=render_config["antialias"],
    )

    shared = sorted(
        cp for cp in set(target.cmap) & set(reference.cmap) if is_han_ideograph(cp, True)
    )
    print(f"{len(shared)} characters are covered by both fonts")

    key = f"topo{SCORE_SIZE}:{target_path.stat().st_mtime}:{reference_path.stat().st_mtime}"
    scores = compute_scores(shared, target, reference, key)

    items = [(cp, scores.get(cp, 0.0)) for cp in shared]
    if args.sort == "suspicious":
        items.sort(key=lambda item: -item[1])

    root = tk.Tk()
    app = Reviewer(root, items, target, reference, args.only_undecided, args.panel)
    if getattr(app, "items", None):
        root.mainloop()


if __name__ == "__main__":
    main()
