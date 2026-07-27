# Third-party notices

## Architectural references only

HanziStyleForge Fusion independently implements ideas described by zi2zi, FontDiffuser, HanziGen, VQ-Font, LF-Font, MX-Font, DeepVecFont-v2 and component-composition research. Their source code, pretrained model weights and font datasets are not included in this repository. See `METHOD_REFERENCES.md` for exact links and scope.

## Optional generation backend: zi2zi-JiT

- Project: `kaonashi-tyc/zi2zi-JiT`
- Upstream: https://github.com/kaonashi-tyc/zi2zi-JiT
- Software license: MIT License, Copyright (c) 2026 Yuchen Tian
- Additional terms for outputs: "Font Artifact License Addendum, Version 1.0"

HanziStyleForge Fusion can optionally delegate glyph generation to zi2zi-JiT
instead of its own generation stack. **Neither zi2zi-JiT's source code nor its
pretrained checkpoints (JiT-B/16, JiT-L/16) are included in or redistributed by
this repository.** You clone the upstream repository and download the
checkpoints yourself, and the backend invokes your local copy. The default
backend remains this project's own implementation; the zi2zi-JiT backend is
opt-in.

The MIT license above governs the zi2zi-JiT software. Its Font Artifact License
Addendum additionally governs generated artifacts and any font product built
from them:

- Commercial and non-commercial use of the artifacts and of font products is
  permitted.
- **If you distribute a font product containing more than 200 unique characters
  created in whole or in part from zi2zi-JiT artifacts, you must provide
  attribution**, including the statement "Created using zi2zi-JiT artifacts" and
  a link to https://github.com/kaonashi-tyc/zi2zi-JiT. The attribution may
  appear in documentation, a README, an about page, package metadata, a
  marketplace listing, or another reasonably visible location distributed with
  the font product.
- At 200 or fewer such characters the Addendum requires no attribution.
- The Addendum grants no trademark rights beyond the attribution text itself.

A typical HanziStyleForge Fusion run rebuilds every Han glyph covered by
`refs/ref.otf`, which is far more than 200 characters. **If you generated with
the zi2zi-JiT backend, assume the attribution requirement applies.** It does not
apply to fonts produced with the default backend, which uses none of zi2zi-JiT's
code, weights or artifacts.

Read the upstream `LICENSE` file for the authoritative text before distributing
anything.

## Redistributed CJK decomposition data

- Project: `amake/cjk-decomp`
- Upstream: https://github.com/amake/cjk-decomp
- Included file: `data/cjk-decomp.txt`
- License option selected for this distribution: Apache License 2.0
- Local notice: `data/NOTICE_CJK_DECOMP.txt`
- Purpose: optional semantic/geometric region hints for local residual retrieval and per-glyph refinement

The decomposition data is not used to decide regional correctness. The actual output structure always comes from the user-supplied `refs/ref.otf`.

## Python dependencies

The program installs third-party packages such as PyTorch, fontTools, Pillow, NumPy, OpenCV and tqdm into the local virtual environment. Each package remains under its own license. The repository does not bundle the virtual environment.

## User-supplied fonts and generated artifacts

This repository does not include `fonts/target.ttf` or `refs/ref.otf`. Apache-2.0 for this software does not grant permission to train on, modify, distribute or sublicense any font. Trained weights and generated fonts may be derivative works of one or both input fonts and remain subject to their licenses.
