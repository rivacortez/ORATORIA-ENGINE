"""Corpus construction: what was recorded, how it is split, and what is frozen.

Phase 1's tooling. `BASELINES.md` §4 already commits this project to three
things, and until now all three were prose:

- *"Speaker-independent. No speaker appears in more than one partition. This is
  the single most common way a speech result becomes meaningless, and the check
  is mechanical rather than a matter of care."* - `split.py` makes it
  mechanical.
- *"The held-out set is frozen at the end of Phase 1 and is not touched during
  tuning."* - `freeze.py` makes "frozen" mean a digest somebody can check
  rather than an intention somebody had.
- *"Dataset cards record provenance, consent basis, recording conditions and
  partition checksums."* - the manifest is the machine-readable half of one.

A promise in a governance document with no code behind it is a promise that
gets kept until the week it is inconvenient.
"""
