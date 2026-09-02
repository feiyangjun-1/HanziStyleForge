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

**Building one.** Two steps, with your eyes in the middle:

```bash
python tools/same_form_review.py render --screen
```

Renders every character both fonts cover as one side-by-side image into
`same_form_review/`, target on the left and ref on the right, named by the
character so a folder in large-icon view is readable as-is. `--screen` drops
the characters whose component, hole or Euler counts already disagree, which
is worth it on a first pass; it is a filter on the workload, not a verdict.

Then delete the images whose two halves differ structurally, and:

```bash
python tools/same_form_review.py collect
```

which reads the filenames that survived and writes this file.

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
