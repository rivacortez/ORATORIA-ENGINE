# Evidence

Raw output from the capture scripts, one file per run.

| Prefix               | Script                        | What it establishes                                |
| -------------------- | ----------------------------- | -------------------------------------------------- |
| `battery-`           | `scripts/capture_evidence.sh` | Every gate, against real Postgres, Redis and MinIO |
| `pilot-a-rehearsal-` | `scripts/rehearse_pilot_a.sh` | The annotators' command line composes end to end   |

A `pilot-a-rehearsal-` file is **not** Pilot A. Pilot A involves two people,
ten minutes of real audio and a question only humans can answer; the rehearsal
runs the part of its checklist a machine can run, so the session with the
annotators is spent on the taxonomy rather than on the tooling. Each capture
says so in its own header and lists what is still outstanding.

These exist because "the integration tests pass" is a claim, and a claim about
a run nobody can point at is worth about as much as no claim.

Every file carries the commit it ran at, **how many files were uncommitted
outside `docs/evidence`**, the Python version, and the unedited output of each
step.

That scope is exact, not shorthand for "the tree was clean": the capture is
being written by `tee` while it counts, so counting itself would report every
clean run as dirty. A capture reporting zero means nothing outside
`docs/evidence` was uncommitted — it says nothing about other captures sitting
beside it.

**The container images are in the `battery-` captures only.** The Pilot A
rehearsal drives a command line over files on disk and starts no
infrastructure, so there is nothing for it to record there, and it does not
pretend otherwise.

They are committed on purpose. A test report that lives in a terminal
scrollback cannot be cited in a thesis, and the integration results are the
ones most easily asserted without being reproduced - they need infrastructure,
so they are the ones that get skipped and then reported as passing.

Regenerate with the stack up:

```bash
docker compose up -d
./scripts/capture_evidence.sh
```
