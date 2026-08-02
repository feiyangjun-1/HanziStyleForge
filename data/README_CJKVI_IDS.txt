Optional CJKVI IDS data
=======================

This folder ships empty on purpose. HanziStyleForge does not redistribute
ids.txt.

The program downloads one pinned revision from
https://github.com/cjkvi/cjkvi-ids and checks it against a recorded SHA-256.
That happens automatically the first time the component atlas or the
component-aware refinement stage needs it.

To fetch it yourself beforehand:

  Windows          .venv\Scripts\python.exe hanzistyleforge.py --config config_months_12gb.json ids-install
  Linux and macOS  .venv/bin/python hanzistyleforge.py --config config_months_12gb.json ids-install

The data supplies hints about which components a character is built from. It
never decides which regional glyph form is correct: the structure of every
rebuilt character comes from the refs/ref.otf you supply.

Review the upstream CHISE and CJKVI license terms before redistributing the
downloaded file.
