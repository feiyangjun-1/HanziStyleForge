# Third-party notices

## Nothing third-party is bundled

This repository contains no upstream source tree, no pretrained weights, no
font files and no font datasets. Everything below is either an idea this
project reimplemented independently, a package installed into the local virtual
environment, or a file downloaded at runtime onto your own machine.

## Architectural references

The architecture was informed by the public work listed here. A citation
records where an idea came from. It does not grant permission to copy code,
weights, datasets or fonts, so check each project's current terms before
reusing anything from it.

| Work | Source | Ideas studied |
|---|---|---|
| zi2zi | https://github.com/kaonashi-tyc/zi2zi | Han glyph style transfer; content and style separation |
| zi2zi-JiT | https://github.com/kaonashi-tyc/zi2zi-JiT | Multi-reference style conditioning; diffusion transformers |
| FontDiffuser | https://github.com/yeungchenwa/FontDiffuser | Diffusion generation; multi-scale content aggregation; explicit style constraints |
| HanziGen | https://github.com/wangwenho/HanziGen | VQ representations with conditional latent diffusion |
| VQ-Font | https://github.com/Yaomingshuai/VQ-Font | Discrete font tokens; structure-aware enhancement |
| LF-Font / MX-Font | https://github.com/clovaai/fewshot-font-generation | Localized component style; factorization; multiple experts |
| DeepVecFont-v2 | https://github.com/yizhiwang96/deepvecfont-v2 | Transformer vector sequences; contour correction |
| Component-composition vector font generation | https://arxiv.org/abs/2404.06779 | Component-region transforms; scalable composition |

## CJK decomposition data, downloaded not bundled

- Project: `cjkvi/cjkvi-ids`
- Source: https://github.com/cjkvi/cjkvi-ids
- Used for: optional Ideographic Description Sequence hints, which help the
  component atlas and per-glyph refinement reason about the parts a character
  is built from

The file is **not** redistributed here. `data/cjkvi-ids/` ships empty. The
program downloads one pinned revision, verified against a recorded SHA-256, the
first time a stage needs it, or when you run the `ids-install` command. Review
the upstream CHISE and CJKVI terms before redistributing the downloaded file
yourself.

The decomposition data never decides which regional glyph form is correct. The
structure of every rebuilt character comes from the `refs/ref.otf` you supply.

## Python packages

Installing puts PyTorch, fontTools, Pillow, NumPy, OpenCV and tqdm into
`.venv`, each under its own license. The virtual environment is not part of
this repository.

## Your fonts, and the fonts this produces

`fonts/target.ttf` and `refs/ref.otf` are yours to provide; neither is included
here. This project's Apache-2.0 license covers this software only. It grants no
right to train on, modify, distribute or sublicense any font.

Trained weights and generated fonts can be derivative works of one or both
input fonts and stay subject to those fonts' licenses. Confirm you have those
rights before distributing anything you produce.
