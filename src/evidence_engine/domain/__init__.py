"""Pure domain model.

No framework, no I/O, no clock read from inside. Contract C2 in
`.importlinter` enforces the first two; the third is a convention the
application layer honours by passing every reading in."""
