# Disagreement log

**Status:** empty — Pilot B has not been run.

Every disagreement between annotators, what was decided, and whether the manual
changed because of it. Committed and diffed, so that "why does the definition
of `self_repair` say that?" has an answer six months from now.

This is a Phase 0 deliverable in its own right: §13's exit criterion requires
that _"unresolved categories are documented"_, and an unresolved category with
no record of the argument is indistinguishable from one nobody noticed.

---

## How to use it

One row per disagreement, in the order the report surfaced them. Fill it
during adjudication, with the audio open.

- **Position** — recording id and millisecond offset, so it can be found again.
- **Ana / Beto** — what each annotated. Class, role, boundaries.
- **Reading** — _why_ each read it that way, in their own words. This is the
  column that matters. "Ana said false_start" is not information; "Ana heard no
  relation between the abandoned fragment and what followed" is.
- **Resolution** — what was decided, or `unresolved`.
- **Manual change** — the taxonomy version this produced, or `none`.

An entry marked `unresolved` is a legitimate outcome and must stay in the log.
§14.4 requires negative and inconclusive results to be preserved, and a
category two trained people cannot agree on is a finding about the taxonomy
that a later reader needs.

---

## Pilot A — technical

Not logged. Pilot A checks the tooling, and disagreements at that stage are
about the interface rather than about the definitions.

Tooling problems found:

| #   | Problem | Fixed in |
| --- | ------- | -------- |
| _   | _       | _        |

---

## Pilot B — taxonomic

### Disagreements

| #   | Position | Ana | Beto | Ana's reading | Beto's reading | Resolution | Manual change |
| --- | -------- | --- | ---- | ------------- | -------------- | ---------- | ------------- |
| _   | _        | _   | _    | _             | _              | _          | _             |

### Unresolved categories

Categories where two trained annotators could not converge, with what makes
them hard. Each one is either a definition to sharpen before the corpus, or a
known limitation to declare in the thesis.

| Category | What the disagreement is about | Decision |
| -------- | ------------------------------ | -------- |
| _        | _                              | _        |

### Boundary conventions settled

Not disagreements about _what_ an event is, but about _where_ it starts and
ends — which is what the boundary-error figures in stage 2 measure. Record any
convention agreed here, because it changes the human ceiling for NFR-004.

| Convention                                                           | Agreed rule |
| -------------------------------------------------------------------- | ----------- |
| Where a filled pause ends when it trails into silence                | _           |
| Whether a cut-off includes the glottal stop                          | _           |
| Where a repetition starts when the repeated word is itself prolonged | _           |

---

## Resulting manual version

|                                     |       |
| ----------------------------------- | ----- |
| Taxonomy version before the pilot   | 1.0.0 |
| Taxonomy version after adjudication | _     |
| Agreement report                    | _     |
| Pilot B re-run after the change     | _     |

> If the taxonomy version changed, Pilot B was re-run under the new manual.
> Reporting agreement measured under a definition that no longer exists would
> attach a number to a document nobody can read.
