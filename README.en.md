[简体中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# HanziStyleForge Fusion

**Regenerate every Han character in one font's style, following another font's glyph standard.**

The tool learns the target font's stroke style, then redraws **every** Han character the reference font covers, following the reference's glyph shapes, and packages the result as an installable `.ttf`.

Note that it regenerates rather than fills gaps: a character already present in the target font is still redrawn to the reference shape. That is what makes the finished set consistent.

Two typical uses:

- **Extend coverage.** The target font has a few thousand characters and you want twenty thousand
- **Change glyph standard.** The target font uses Taiwan forms (or Japanese, or Korean) and you want mainland Chinese ones. Swap in a mainland reference font: the style is kept, the shapes are redrawn to the new standard

> Experimental. A full run takes days to weeks. You can interrupt it at any point and pick up where you left off.

---

## You provide two fonts

| File | Role | What it means |
|---|---|---|
| `fonts/target.ttf` | **Style** | The font you like. Only its stroke appearance is learned |
| `refs/ref.otf` | **Shapes** | A font with full character coverage. Its glyph structure is followed |

For example: put a handwriting-style font at `target.ttf` and Source Han Sans at `refs/ref.otf`, and you get that handwriting style drawn to Source Han Sans' character shapes.

**`ref.otf` decides the glyph standard.** Want mainland Chinese forms? Use a mainland reference font. Want Taiwan, Hong Kong, Japanese or Korean forms? Swap it for one of those. The program never decides which is "more correct" on its own.

### What the font files must be

- Static fonts. **Variable fonts, TTC and OTC are not supported**
- `target.ttf` must be TrueType (with a `glyf` table)
- `ref.otf` may be TrueType or CFF/OTF

---

## Hardware

**An NVIDIA GPU is required.** Training is the bulk of this project and needs CUDA.

| Platform | Can it train? |
|---|---|
| Windows + NVIDIA | Yes |
| Linux + NVIDIA | Yes |
| macOS | **No.** Macs have no NVIDIA GPU, Apple's MPS is not supported here, and CPU-only training is impractically slow |
| Linux or Windows without a discrete GPU | Same as above |

Also: Python 3.10-3.14, and at least 150 GB of free disk recommended. 12 GB of VRAM is enough; the shipped configuration is tuned for it.

On a Mac you can still install, run the self-test, inspect fonts, or build a font from glyph images you already have. For training, use a machine with an NVIDIA GPU.

---

## Getting started

### Windows

Double-click these four, in order:

| Step | Double-click | What it does |
|---|---|---|
| 1 | `install_cuda130.bat` | Sets up the environment. Once only |
| 2 | — | Put your two fonts in `fonts\` and `refs\` |
| 3 | `verify_project.bat` | Checks everything is in place |
| 4 | `run_months_resilient.bat` | Starts the run |

To pause, double-click `request_safe_stop.bat`. The run exits safely at its next checkpoint. Double-click `run_months_resilient.bat` again to continue.

### Linux and macOS

Open a terminal in the project folder:

```bash
./install.sh
```

Put your two fonts in `fonts/` and `refs/`, then:

```bash
./verify.sh
./run.sh
```

To pause:

```bash
./stop.sh
```

Run `./run.sh` again to continue from the checkpoint.

> If you get `Permission denied`, run `chmod +x *.sh` first.

---

## Where the output goes

```text
build/target-HanziStyleForge-Fusion.ttf               ← the font
build/target-HanziStyleForge-Fusion.ttf.report.json   ← build report
work_hanzistyleforge_fusion_months/qa/index.html      ← QA report, open in a browser
```

**Read the QA report before installing the font.** It shows every glyph side by side with the reference and the target, so you can see which characters came out well and which fell back to the reference shape.

Training data, checkpoints and generation progress live in `work_hanzistyleforge_fusion_months/`. It runs to tens of gigabytes. **Do not delete it while a run is in progress.**

---

## What happens if it gets interrupted

Nothing bad. Every stage and every generated glyph is checkpointed. Run the same command again and it continues from where it stopped.

That covers power cuts, crashes and Ctrl+C alike. `run_months_resilient.bat` and `run.sh` also retry automatically after an error, giving up only after 20 consecutive failures, which means the fault is real rather than a passing hiccup.

---

## Questions people ask

**How long does it take?**
Days to weeks, depending on character count and GPU. On a 12 GB laptop GPU with the shipped configuration, plan in weeks.

**Can I do just some characters?**
Yes. Set `scope.mode` to `chars_file` in the configuration and point `scope.extra_chars_file` at a list, one character or `U+4E00`-style codepoint per line. Trying a few hundred characters first is a good habit.

**Could it produce characters that break the glyph standard?**
There is a structural gate: if a generated glyph's connected-component count or hole count disagrees with the reference, it is rejected and the reference shape is used instead. That prevents characters with a stroke too many or too few, at the cost of some characters keeping the reference look rather than your target style.

**Does it touch anything that is not a Han character?**
No. Latin letters, digits, punctuation, kana, Hangul, and the OpenType layout and hinting tables are carried over from `target.ttf` unchanged, and verified byte for byte during the build.

**I get `requires CUDA, but torch.cuda.is_available() is False`.**
No usable NVIDIA GPU was found. Update your driver, or check that you installed a CUDA build of PyTorch.

**I get `does not support training.device='mps'`.**
Apple GPUs are not supported. See the hardware section above.

---

## Before you use it

- A full run may take days, weeks or longer
- This repository contains no fonts, pretrained weights or third-party datasets
- **A generated font may be subject to both the `target.ttf` and `ref.otf` licenses.** Use only fonts you have the right to train on, modify and distribute
- This is experimental software. Check the QA report and test the font yourself before releasing anything

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for references and licensing.

---

## How it works

```text
target.ttf (style)   ref.otf (glyph structure)
        └──────┬──────┘
               ↓
   style encoder → VQ stroke codebook → latent diffusion → refiner
               ↓
   multiple candidates → structural gate → QA → outline tracing → TTF
```

Style is learned only from `target.ttf` and structure taken only from `ref.otf`. The two data paths are kept separate and that separation is verified at runtime.

## Contributing

Issues and pull requests are welcome. When contributing third-party code, data or models, include the source and license.
