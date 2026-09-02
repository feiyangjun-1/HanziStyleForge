# data/

## same_form_han.txt (not shipped -- you supply this)

`analysis.same_form_chars_file` points here. It is a plain-text list of the
characters for which your reference font and your target font draw the *same
structural form* -- the same components in the same arrangement, differing only
in style. `prepare` turns each one into a `mode="cross"` training row that pairs
the reference's structure proxy with the target's real glyph, which is the only
place the model is shown a genuine reference-to-target example.

A character whose two fonts disagree structurally must not be in this list: the
pair would teach the model to convert one regional form into another, which is
the opposite of what the reference is for.

**Format**: the characters themselves, any amount of whitespace or line breaks
between them; `U+XXXX` tokens and `#` comment lines are also accepted.

**Building one.**

```bash
python tools/same_form_review.py
```

Opens a window showing each character in both fonts side by side. Press `y` if
the forms match and `n` if they do not; `left`/`right` move without deciding and
`u` clears a verdict. Every keystroke rewrites this file, so closing the window
or losing power costs at most the keystroke in flight, and reopening resumes at
the first undecided character. `--only-undecided` skips what you have already
judged, and `--sort suspicious` puts the largest topology disagreements first,
which clears the obvious mismatches quickly.

If this file already exists when no progress file does, its characters are
loaded as accepted verdicts and a `.bak` copy is kept, so an existing list is
refined rather than overwritten.

**Automatic screening is not sufficient on its own.** Comparing
two styled fonts directly does not work at all -- stroke weight and terminal
shape swamp the signal, and no threshold separates them. Comparing two regional
cuts of one family (for example Source Han Serif CN against TW) is much cleaner,
because the region is then the only variable. But every such check is a topology
comparison, and component, hole and Euler counts cannot see a glyph drawn in a
Japanese form, or one with the wrong stroke terminal, or a truncated stroke.
Those pass silently. Screen automatically to narrow the field if you like, then
look at every survivor rendered beside its counterpart before trusting it.

If the file is absent no cross rows are built and training falls back to target
self-reconstruction alone, which still runs but gives the model no example of
the input distribution it actually meets at generation time.
