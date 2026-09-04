# Evidence

Raw output from `scripts/capture_evidence.sh`, one file per capture.

These exist because "the integration tests pass" is a claim, and a claim about
a run nobody can point at is worth about as much as no claim. Each file carries
the commit it ran at, whether the tree was clean, the container images, and the
unedited output of every gate.

They are committed on purpose. A test report that lives in a terminal
scrollback cannot be cited in a thesis, and the integration results are the
ones most easily asserted without being reproduced - they need infrastructure,
so they are the ones that get skipped and then reported as passing.

Regenerate with the stack up:

```bash
docker compose up -d
./scripts/capture_evidence.sh
```
