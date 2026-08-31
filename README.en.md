[简体中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# HanziStyleForge

Regenerate every Han character in one font's style, following another font's glyph standard.

The tool learns stroke style from a target font, redraws every Han character the reference font covers using the reference's structure, and packages the result as an installable `.ttf`. Characters the target font already has are redrawn too, so the whole set stays consistent.

Two typical uses:

- Extend coverage. A target font with a few thousand characters, extended to over twenty thousand
- Change glyph standard. A target font drawn to Taiwan (or Japan, Korea) forms, redrawn to mainland forms by swapping in a mainland reference, style preserved

> Experimental. A full run takes days to weeks and can be interrupted and resumed at any point.

---

## What you need

### Two fonts (required)

| File | Role |
|---|---|
| `fonts/target.ttf` | Style source. The only thing taken from it is what strokes look like |
| `refs/ref.otf` | Structure source. Its glyph shapes are followed, and its coverage decides which characters are generated |

Matched by filename stem, extension is free: `refs/ref.ttf` works just as well.

The glyph standard is whatever `ref` uses. Put a mainland font there for mainland forms, swap it for Taiwan, Hong Kong, Japan or Korea standards. The program does not decide which is more correct.

Static fonts only; variable fonts, TTC and OTC are unsupported. `target` needs a `glyf` table (TrueType), `ref` can be TrueType or CFF/OTF.

### Same-form character list (optional, affects quality)

`data/same_form_han.txt`, not shipped, you supply it. It lists the characters your two fonts draw with the **same structure** -- same components in the same arrangement, differing only in style.

With this file, `prepare` builds one "reference structure to target glyph" training sample per character. That is the only time the model sees a real reference glyph as input. Without it, training is target self-reconstruction alone, which runs, but the model never sees the input distribution it actually meets at generation time.

Characters whose two fonts disagree structurally **must not** be in the list. Such a pair teaches the model to convert one regional form into another, which is the opposite of what the reference is for.

Format and how to build one are in [data/README.md](data/README.md). In short: automatic screening narrows the field, but a topology comparison cannot see a Japanese glyph form or a wrong stroke terminal, so every survivor still has to be looked at.

---

## Hardware

An NVIDIA GPU is required; training does not run without CUDA.

| Platform | Training |
|---|---|
| Windows / Linux + NVIDIA | Yes |
| macOS | No. No NVIDIA GPU, and Apple MPS is unsupported |
| No discrete GPU | No |

Python 3.10-3.14, 150 GB free disk recommended. 12 GB of VRAM is enough; the defaults are tuned for it.

On a Mac you can still install, self-test, inspect fonts, or package already-generated glyph images into a font. Training needs a machine with an NVIDIA card.

---

## Getting started

### Windows

| Step | Double-click | What it does |
|---|---|---|
| 1 | `install.bat` | Install the environment, once |
| 2 | — | Put the fonts in `fonts\` and `refs\` |
| 3 | `verify.bat` | Check |
| 4 | `run.bat` | Run |

`stop.bat` exits safely at the next checkpoint; double-click `run.bat` to resume.

### Linux / macOS

```bash
./install.sh          # install the environment
./verify.sh           # check, once the fonts are in place
./run.sh              # run; ./stop.sh to pause
```

Run `chmod +x *.sh` first if you get `Permission denied`.

---

## Output

```text
build/target-HanziStyleForge.ttf              the font
build/target-HanziStyleForge.ttf.report.json  build report
work/qa/index.html                            QA report
```

Read the QA report before installing. It has per-character comparisons showing which characters generated well and which fell back to the reference glyph.

Training data, checkpoints and generation progress all live in `work/`. It runs to tens of gigabytes. Do not delete it mid-run.

---

## VRAM settings

The default `config.json` is tuned for 12 GB and needs no changes on a 12 GB card. Peak single-step usage measured on an RTX 5070 Ti Laptop (11.9 GB):

| Stage | Default batch | Peak VRAM |
|---|---|---|
| vq256 | 4 | 3.1 GB |
| vq384 | 2 | 3.4 GB |
| vq512 | 1 | 3.0 GB |
| direct256 | 6 | 4.6 GB |
| direct384 | 3 | 5.2 GB |
| latent256 | 6 | 5.4 GB |
| latent384 | 4 | 7.8 GB |
| latent512 | 4 | 11.2 GB |

### Recommended

| VRAM | Change |
|---|---|
| 16 GB+ | None |
| 12 GB | None |
| 10 GB | `latent512`: `batch_size` 4 to 2, `gradient_accumulation` 1 to 2 |
| 8 GB | The above, plus `latent384` 4 to 2 / 1 to 2 |
| 6 GB | Halve `batch_size` and double `gradient_accumulation` everywhere |

The rule: keep `batch_size` times `gradient_accumulation` constant, or the effective batch changes and the learning rate no longer matches it.

More VRAM does not make it much faster. The bottleneck is compute, not memory: going from batch 4 to 8 gained under 10% throughput.

---

## CPU settings

`training.workers` is the number of data-loading processes, 4 by default.

| Physical cores | workers |
|---|---|
| 4 | 2 |
| 6-8 | 4 (default) |
| 12+ | 6-8 |

Each worker costs roughly 1.5-2 GB of RAM. Lower it if memory is tight.

Do not raise `cpu_threads`, `interop_threads` or `opencv_threads`. Training is GPU-bound, and raising these only lets the CPU compete with GPU scheduling.

Whether more workers would help: watch batch/s in the terminal. If GPU utilization (`nvidia-smi`) sits below 80%, more workers help; at 90% or above it is compute-bound and they will not.

---

## Editing the configuration

`config.json` in the project root; all four launchers pass it. Save as UTF-8.

Every stage has a `name` field, so search by name. For an 8 GB card, find `fusion` then `diffusion` then `phases`, and the entry with `"name": "latent512"`:

```json
{
  "name": "latent512",
  "batch_size": 2,
  "gradient_accumulation": 2
}
```

The product is still 4, unchanged.

### What invalidates existing progress

Checkpoints are matched by fingerprint. A mismatch restarts that stage from epoch 1.

| Change | Effect |
|---|---|
| Anything in `fusion.vq.phases[]` | That VQ stage retrains |
| Anything in `fusion.diffusion.phases[]` | That diffusion stage retrains |
| `fusion.style_encoder`: `size`, `epochs`, `batch_size`, `learning_rate`, `virtual_length`, `references_per_set`, `cell_grid`, `query_gain` | Style stage retrains |
| `fusion.refiner`: `size`, `epochs`, `batch_size`, `gradient_accumulation`, `learning_rate` | Refiner retrains |
| `fusion.style_encoder.early_stopping` | Safe |
| `fusion.refiner.minimum_epochs`, `minimum_relative_improvement` | Safe |
| `training.workers`, `training.amp`, `checkpoint_every_steps`, `preview_every` | Safe |
| Anything in `build` | Safe, read only when the TTF is built |

The VQ and diffusion `early_stopping` blocks live inside their phases, so changing them does retrain that stage. Style and refiner differ here.

Settle VRAM-related settings before starting. Changing one mid-run costs that stage.

### Out of memory

Note which stage the error came from (the terminal prints the stage name) and change only that one: halve `batch_size`, double `gradient_accumulation`, repeat if needed.

### The font is too large

Almost all of the size is glyph outlines. `build.curve_simplify` controls it -- larger values keep fewer points. Measured on the reference font across 17 characters of varying complexity:

| `curve_simplify` | Points/glyph | Dice |
|---|---|---|
| 0.55 | 611 | 0.9848 |
| 1.2 (default) | 167 | 0.9843 |
| 1.8 | 130 | 0.9812 |
| 2.5 | 117 | 0.9786 |

A hand-designed font runs about 156 points per glyph, so the default sits at that level: roughly 14 MB for twenty thousand characters. Past 1.8 the fit degrades noticeably. Going down to 0.55 gains nothing either, since the extra points trace raster aliasing rather than the glyph.

The other knobs are weak. Dropping `maximum_points_per_contour` from 480 to 96 only reached 69% of the size; leave it alone.

To see where the size went:

```bash
python tools/font_size_report.py build/target-HanziStyleForge.ttf
```

### Checking your edits

```bash
verify.bat        # Linux/macOS: ./verify.sh
```

Checks JSON syntax and value ranges. The most common JSON mistake is a missing or extra comma; the last entry must not have one.

---

## Interruptions

Every stage and every generated glyph is checkpointed, so running the same command again resumes from where it stopped. Power loss, a blue screen and Ctrl+C are all the same. The launchers also retry after an error and give up only after 20 consecutive failures.

If you resume by typing commands rather than using a launcher: requesting a stop leaves a `STOP_AFTER_CHECKPOINT` marker in the project root. The launchers delete it on every start, typed commands do not, and a new run would stop at its first checkpoint. Delete it before resuming manually.

---

## Other tools

```bash
python png_to_svg.py <image-dir> [output-dir]
```

Vectorizes a directory of glyph images into SVG using the same outline extraction as `build`. The simplification tolerance scales with the input image size.

---

## FAQ

**How long does it take?** Days to weeks, depending on character count and GPU. The defaults on a 12 GB laptop GPU are a matter of weeks.

**Can I run only some characters?** Yes. Set `scope.mode` to `chars_file` and point `scope.extra_chars_file` at a list (one character or one `U+4E00`-style codepoint per line). Trying a few hundred first is a good habit.

**Will generated characters be malformed?** There is a structure gate: a candidate whose component and hole counts disagree with the reference is rejected and the reference glyph is used instead. That prevents characters with a stroke too many or too few, at the cost of some characters keeping the reference's look.

**Is non-Han content modified?** No. Latin, digits, punctuation, kana, hangul, and OpenType layout and hinting data are all preserved byte-for-byte from `target.ttf` and verified during the build.

**`requires CUDA, but torch.cuda.is_available() is False`** No usable NVIDIA GPU was found. Update the driver, or check that the CUDA build of PyTorch is installed.

**`does not support training.device='mps'`** Apple GPUs are unsupported.

---

## Before you use it

- A full run may take days to weeks
- This repository contains no font files, pretrained weights or third-party datasets
- The generated font may be bound by the licenses of both `target.ttf` and `ref`. Only use fonts you have the right to train on, modify and publish
- This is experimental. Check the QA report and test by hand before publishing

Licensed under Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The license covers the software only and grants no right to train on, modify or publish any font. Third-party references are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## How it works

```text
target.ttf (style)   ref.otf (structure)
        └──────┬──────┘
               ↓
   style encoding → VQ stroke codebook → latent diffusion → refiner
               ↓
   multi-candidate generation → structure gate → QA → vectorization → TTF
```

Training samples come in two kinds: target self-reconstruction, and the "reference structure to target glyph" pairs covered by the same-form list. Ground truth in both comes only from `target.ttf`, structure input comes only from `target` itself or from `ref`, and the data flow is verified at runtime.

## Contributing

Issues and pull requests are welcome. When submitting third-party code, data or models, state the source and license.
