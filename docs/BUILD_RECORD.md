# Build record

What was built, in what order, and why each decision went the way it did.

This is the parent document. The ADRs record *architectural* decisions in
isolation; this records the build itself — including the decisions that were
wrong and got reversed, which is the part a thesis defence actually needs. A
record that only lists successes cannot be checked against anything.

**How to read it.** Every claim here should be verifiable from the repository:
a commit, a file, a test, or a capture under `docs/evidence/`. Where a claim
cannot be verified, it says so.

---

## Contents

1. [The shape of the work](#1-the-shape-of-the-work)
2. [What is actually done](#2-what-is-actually-done)
3. [Decisions, and the reasoning that survived review](#3-decisions-and-the-reasoning-that-survived-review)
4. [Mistakes, and what they cost](#4-mistakes-and-what-they-cost)
5. [What is blocked, and by what](#5-what-is-blocked-and-by-what)
6. [Phase map: 3 through 10](#6-phase-map-3-through-10)
7. [Evidence index](#7-evidence-index)

---

## 1. The shape of the work

The engine is an independent external service, built from zero, separate from
the `ORATORIA-PROYECTO` monolith that consumes it (ADR-001). Clean/hexagonal
architecture with the dependency rule machine-checked by eight `import-linter`
contracts, so an architectural claim in a document fails CI before it reaches a
reader.

Two trees live in one repository and are kept apart by contract:

| Tree                 | What it is                                              | Contract |
| -------------------- | ------------------------------------------------------- | -------- |
| `src/evidence_engine` | the deployable service                                  | C1–C6    |
| `src/corpus`          | research tooling — annotation, agreement, corpus building | C7, C8   |

C7 lets the corpus import `evidence_engine.domain.shared` and nothing else;
C8 forbids the service from importing the corpus at all. A deployed engine
must not need an annotation parser, and research tooling must not depend on
the service's transaction handling.

---

## 2. What is actually done

| Phase | Deliverable                                | State |
| ----- | ------------------------------------------ | ----- |
| 0     | Scope, taxonomy, manual, consent, ADRs     | deliverables done; exit criterion needs two human annotators |
| 0.5   | Experimental closure of the **speech** taxonomy | tooling done and reviewed four times; pilots not run |
| 1     | Corpus construction                        | tooling done; nothing recorded |
| 2     | Platform skeleton and contracts            | done, exit criterion proven on real infrastructure |
| 3–10  | see [§6](#6-phase-map-3-through-10)        | |

### Phase 2 — the platform

Closed and proven. Sessions, transcript, speech events, visual events, quality
and evidence as separate domain packages; an application layer that talks only
to ports; adapters for REST, WebSocket, PostgreSQL, Redis, S3/MinIO and two
deterministic model runtimes. Verified against real Postgres, Redis and MinIO,
not only in memory — three bugs were invisible in memory and only appeared
against the real thing: a missing `processing_run` parent row, a partial unique
index violated by flush ordering, and an object-store adapter with no
credentials.

### Phase 0.5 — the annotation and agreement tooling

ELAN as the annotation surface, configured rather than an editor written here.
Controlled vocabularies are generated from the published taxonomy, so an
annotator cannot type a class that is not published, and the manual and the
tool cannot disagree.

The agreement calculator answers three separate questions in order, because
they have three different fixes:

1. **Did they find the same events?** Positive specific agreement over an
   optimal non-crossing matching. Never counts a true negative.
2. **Did they draw the same boundaries?** Median and p95 start/end error over
   matched pairs — the human ceiling for NFR-004's 250 ms target, which nothing
   had measured.
3. **Did they give them the same label?** Cohen's κ and nominal Krippendorff's
   α over matched events only, plus the per-class confusion matrix.

Plus an event-by-event adjudication worklist: every position that needs
discussing, in recording order, with a seekable timestamp and both readings,
split into missed / class / role / boundary.

### Phase 1 — corpus construction

`corpus inventory` (is there enough of each class from enough different
speakers), `corpus split` (speaker-independent, stratified) and `corpus split
--freeze` / `corpus verify` (the held-out manifest with per-file digests).

---

## 3. Decisions, and the reasoning that survived review

Each of these was challenged. What is recorded is the reasoning that held.

### 3.1 Agreement is computed over matched events, never over time frames

**Decision.** `measures.kappa_over_time_frames` exists and does nothing but
raise `UndefinedUnitError`.

**Why.** Discretise a recording into 10 ms frames, label each with the class
covering it or "none", compute Cohen's κ: about 95% of frames are "none", both
annotators agree on essentially all of them, and κ comes back near 0.9 while
telling you nothing. The coefficient measures the silence. The function exists
so the mistake has somewhere to fail loudly instead of somewhere to succeed
quietly.

### 3.2 The matching is optimal and non-crossing, not greedy

**Why non-crossing.** Events sit on a timeline. A match where A's third event
pairs with B's fifth while A's fourth pairs with B's second asserts that two
events swapped places in time.

**Why optimal.** Greedy is order-dependent and a reviewer can reasonably ask
what it missed. Under the non-crossing constraint the optimum is a dynamic
program in O(nm) — the same shape as sequence alignment — short enough to check
by reading.

### 3.3 The same matching rule scores the model

**Why.** It puts the human ceiling and the model's F1 on one scale. If two
trained annotators reach 0.78 positive specific agreement on `false_start`, a
model reporting 0.80 is at the ceiling, and NFR-002's 0.80 target has to be
read against 0.78 rather than against 1.0.

### 3.4 Everything that would produce a meaningless number is refused, not noted

**Decision.** `compare()` refuses: files the validator rejected, an adjudicated
pass, two files by one annotator, and any difference in schema or taxonomy
version.

**Why.** Every one of these produces a report that renders perfectly — the
coefficients compute, the confusion matrix fills in — and is about something
other than what the reader will take it to be about. A note at the bottom of a
report does not survive being copied into a results table.

### 3.5 Phase 0 has two halves and only one is being closed

**Decision.** Phase 0.5 validates the nine speech classes. The nine visual
classes stay published and unvalidated until Phase 5, and every document that
claims a closure has to name which half.

**Why.** The pilots never show an annotator a video frame. Also: a visual event
has none of a speech event's relationships (no words tier to check against, no
contextual role, no lexical expression), and `insufficient_lighting` is a
continuous condition rather than an event — two annotators marking it over
overlapping stretches are unitizing, not detecting, and the one-to-one
non-crossing matching is the wrong instrument for that.

Enforced by `tests/corpus/test_scope.py`, which sweeps the claim-bearing
documents for closure phrasings that must not appear.

### 3.6 Partitions are by speaker and stratified

**Why by speaker.** Assigning recordings and hoping no speaker straddles a
boundary is the failure mode itself: a model that has heard a speaker in
training and is scored on that speaker in held-out is being scored on
memorisation. Speakers are assigned; recordings follow.

**Why stratified.** On a corpus of tens of speakers with nine classes, random
assignment routinely lands zero instances of a rare class in held-out. Its
per-class F1 is then *undefined* — a missing measurement that reads as a low
score.

### 3.7 Adequacy is a pair, not a count

**Decision.** A class is inadequate unless it has enough instances **and**
comes from enough different speakers.

**Why.** Forty `cut_off` from one person is that person's habit. The protocol
already says agreement measured on one speaker is agreement about that speaker;
a per-class F1 is no different.

### 3.8 "Frozen" means a digest, not an intention

**Why.** §14.4 makes the held-out freeze a scientific gate, and a gate enforced
by everyone remembering is not a gate. The failure it exists to prevent —
regenerating the held-out set after seeing a disappointing number — looks
exactly like ordinary work while it happens and is undetectable afterwards.

### 3.9 The baseline pins name artifacts, not runs

**Decision.** Four freeze stages, not two:

| Frozen | When |
| --- | --- |
| Model artifacts and runtime-independent decoding configuration | done, Phase 0.5 |
| The held-out set | end of Phase 1 |
| The executable environment (backend, container, Torch/CUDA) | Phase 3 |
| The baseline outputs over the held-out set | Phase 3, **after** the environment |

**Why the last two are in that order.** A pinned weights digest guarantees the
same weights, not the same numbers: ct2 and transformers do not produce
bit-identical output from the same weights, and neither does one backend across
two CUDA builds. Generating outputs as soon as the held-out set exists would
produce them on whatever backend happened to be installed.

### 3.10 87.8 is not the NFR-001 threshold

**Decision.** CrisperWhisper 2.0's published disfluency-F1 average is context.
The comparator is the same baseline run over the human-annotated `es-PE`
held-out set, which does not exist yet — so NFR-001 correctly has no number.

**Why.** 87.8 averages ten languages of which eight, Spanish included, are
evaluated on **synthetic** verbatim sets, and it includes English and German,
which this thesis does not measure.

---

## 4. Mistakes, and what they cost

Recorded because a build record that only lists successes cannot be checked,
and because each of these produced a rule that is now enforced.

### 4.1 Pinned the wrong model

CrisperWhisper **v1** was pinned as the verbatim comparator for a Peruvian
Spanish corpus. Its own model card says *"CrisperWhisper 1.0 is
English/German-only"*, with `language: ['de','en']` in the metadata and every
cited benchmark English. The choice was defended in writing on the grounds of
not swapping baselines mid-project — a sound argument on an unchecked premise.

**Cost.** One review round. **Rule.** A pin is only as good as the reading of
the model card that produced it; the card's own language claim is checked and
quoted in `BASELINE_PINS.md`.

### 4.2 Destroyed `.gitattributes`

Adding one rule used `cat >` instead of `cat >>` and replaced the file,
silently dropping repository-wide LF normalization, `tests/contract/golden/**
-text` (byte-for-byte fixture comparison) and the media binary declarations.

**Cost.** Nothing yet — it was caught in review. It would have surfaced months
later as a contract test that passes on one machine and fails on another.

**Rule.** Never overwrite a file that has not been read. `test_scope.py` now
asserts every rule by literal string, because the loss is invisible in a diff
that shows a file rewritten wholesale: the reviewer sees new content that reads
as reasonable and has no reason to reconstruct what was there.

### 4.3 Tests asserted presence, never absence

`test_scope.py` checked that the *correct* phrasing existed in each document
and never that the *contradictory* phrasing was gone. Three unqualified closure
claims survived four review rounds that way — including `PILOT_PROTOCOL.md`'s
own title.

**Rule.** A document can carry the qualified statement in one section and the
unqualified one in another. The claim-bearing documents are now swept for
phrasings that must not appear anywhere in them. The sweep reads *assertions*,
stripping quoted and backticked spans first — two documents legitimately quote
the forbidden phrasing inside the sentence that forbids it, and a check that
cannot tell a claim from its prohibition demands deleting the warning.

### 4.4 Reported a number I had not read

An evidence commit message said 890 tests against a capture that said 894. The
history was already pushed and the force-push was refused, so the wrong number
is permanent in that commit; the capture is authoritative and the discrepancy
is recorded in the PR.

**Rule.** Read the figure out of the capture before writing it anywhere.

### 4.5 Stratification scored the wrong signal

The first partition scorer ranked partitions by their *largest* outstanding
class need. `filled_pause` outnumbers `cut_off` roughly ten to one, so the
common class dominated every comparison and the rare one — the only class whose
placement is actually at risk — had no influence at all. The partitions came
out proportional in bulk and short of `cut_off` in dev: precisely the outcome
stratification exists to prevent.

**Cost.** Caught by the coverage test, not by inspection. **Rule.** Score on
the speaker's *rarest* class.

### 4.6 Order-dependent matching

The dynamic program broke equal-optimal ties with `best[i][j] == best[i-1][j]`
— "prefer the left side" — so the confusion matrix, the boundary errors and
every per-class figure depended on which annotator's file was passed first.

The counterexample was found by brute-force search and shrunk to four
annotations; **the randomised property tests pass against the buggy code**,
because the backtrack is naturally symmetric except at an exact tie between two
different optimal matchings, and independently generated annotations
essentially never produce one. Catching it needed a generator drawing intervals
from a tiny repeated grid.

**Rule.** Verify a fix by reintroducing the defect and requiring the suite to
turn red. Every code fix in rounds two, three and four was checked that way.

---

## 5. What is blocked, and by what

Four things gate almost everything downstream. None of them is an engineering
problem.

| Blocker | Blocks | Unblocked by |
| --- | --- | --- |
| **No recorded corpus** | Phases 1 (execution), 4, 6 (evaluation), 9 | recording speakers |
| **No human annotators** | Phase 0.5's exit criterion, Pilots A and B | two trained people |
| **No model weights, torch not a dependency** | Phase 3, 4 | downloading the pins and adding the runtime |
| **No provisioned inference host** | NFR-005/006/007, Phase 7's figures | provisioning one |

To start Pilot A, four audio confirmations and nothing else: recording from the
named physical microphone, mono PCM, a declared sample rate and bit depth, and
NVIDIA Broadcast and Voicemeeter out of the path — confirmed from the recorded
file rather than a settings dialog, because a dialog reports what was requested
and the file reports what happened.

---

## 6. Phase map: 3 through 10

_Filled from the phase-mapping run. See §7 for the evidence._

---

## 7. Evidence index

Raw, unedited captures under `docs/evidence/`, each carrying the commit it ran
at and how many files were uncommitted outside `docs/evidence`.

| Prefix | Script | What it establishes |
| --- | --- | --- |
| `battery-` | `scripts/capture_evidence.sh` | every gate, against real Postgres, Redis and MinIO |
| `pilot-a-rehearsal-` | `scripts/rehearse_pilot_a.sh` | the annotators' command line composes end to end |

A `pilot-a-rehearsal-` capture is **not** Pilot A. It runs the part of Pilot
A's checklist a machine can run, so the session with the annotators is spent on
the taxonomy rather than on the tooling.
