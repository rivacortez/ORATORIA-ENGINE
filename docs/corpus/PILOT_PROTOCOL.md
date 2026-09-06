# Phase 0.5 — Experimental closure of the speech taxonomy

**Status:** protocol approved, not yet executed
**Blocks:** Phase 1 (corpus construction)

---

## Why this phase exists

§13 separates Phase 0's _deliverables_ from its _exit criterion_:

> Two annotators can apply the taxonomy consistently to a pilot sample and
> unresolved categories are documented.

The deliverables are done — the taxonomy is frozen, the manual is generated
from it, the policies are written, the ADRs are recorded. The criterion is not:
it requires two humans annotating a pilot sample, and that has not happened.

Recording a corpus before that measurement is the most expensive mistake
available in this project. If forty speakers are recorded and annotated, and it
then turns out that annotators systematically split `false_start` from
`self_repair` differently, the corpus is inconsistently labelled and every
per-class F1 computed on it measures annotator noise instead of model
performance. The corpus cannot be repaired without re-annotating it, and
re-annotation costs the same as the original.

So the speech taxonomy gets closed experimentally first, with a small sample,
and the manual gets whatever corrections the disagreements demand. The visual
half is not touched here; see the scope section immediately below.

## Scope: the speech taxonomy only

**Phase 0.5 validates the nine speech classes. It says nothing about the nine
visual ones.** This is a deliberate narrowing, stated here because the phrase
"each P0 class" below would otherwise be read as covering the whole taxonomy,
and a reviewer is entitled to know exactly what was measured.

What is in scope: `filled_pause`, `lexical_filler`, `repetition`,
`false_start`, `self_repair`, `cut_off`, `prolongation`, `silent_pause`,
`unintelligible`, and the contextual roles attached to the lexical classes.

What is not: `gaze_toward_camera`, `gaze_away_from_camera`,
`face_visibility_loss`, `insufficient_lighting`, `insufficient_framing`,
`posture_deviation`, `repetitive_torso_movement`, `self_touch`,
`repetitive_hand_movement`.

Three reasons, in order of weight.

_The annotation record has no field for them._ `corpus.schema.records` models
a speech event: an interval, a class, a contextual role, a raw expression, and
the words tier they are checked against. A visual event has none of those
relationships and would need a second record shape, a second validator and a
second set of agreement conventions — a body of work comparable to this whole
phase.

_Visual agreement is a different measurement._ `insufficient_lighting` is a
continuous condition, not an event: two annotators marking "the lighting was
bad" over overlapping stretches are performing unitizing, not detection, and
the matching here — one-to-one, non-crossing, IoU-thresholded — is the wrong
instrument for it. Reporting a visual kappa produced by this tooling would be
a number about the tooling.

_The expensive mistake is on the speech side._ Phase 1 records and annotates
audio. The visual channel is annotated in Phase 5, when the corpus already
exists, and a taxonomy error there costs re-annotating one modality rather
than re-recording the corpus.

**Consequence for the thesis.** Any claim of the form "two annotators applied
the taxonomy consistently" must read "applied the _speech_ taxonomy
consistently". The visual taxonomy enters Phase 5 unvalidated, and closing it
needs its own pilot with its own protocol — including deciding whether its
classes are events at all.

This scope is machine-checked: `tests/corpus/test_scope.py` fails if the
annotation schema learns to carry a visual class without this section being
rewritten.

## Two pilots, not one

Ten minutes of audio can tell you whether the tooling works. It cannot tell you
whether the speech taxonomy is consistent, because ten minutes contains perhaps three
examples of the rarer classes and agreement on three examples is not evidence
of anything.

### Pilot A — technical

**Purpose:** the tool, the format and the pipeline, not the taxonomy.

|            |                                                                               |
| ---------- | ----------------------------------------------------------------------------- |
| Material   | ~10 minutes, one speaker, from a recording that will **not** enter the corpus |
| Annotators | both                                                                          |
| Question   | Does the round trip work end to end?                                          |

Checks:

- The ELAN template opens, the controlled vocabularies appear, and no class
  outside the taxonomy can be selected.
- `corpus validate` reads both files without error.
- `corpus agreement` produces a report.
- The annotators can explain, in their own words, what each tier is for.

**Exit:** both files parse, the report renders, and neither annotator is
guessing at the interface. Agreement numbers from Pilot A are **not** reported —
the sample is too small and the annotators are still learning the tool, so
whatever they show is about the tool.

### Pilot B — taxonomic

**Purpose:** whether the speech taxonomy is consistently applicable.

|            |                                                                         |
| ---------- | ----------------------------------------------------------------------- |
| Material   | several speakers, sampled to contain enough of each P0 **speech** class |
| Annotators | both, independently and blind                                           |
| Question   | Do two trained people apply these definitions the same way?             |

**Sampling is by class, not by duration.** The requirement is "enough examples
per speech class", and the rarer classes drive the sample size: `cut_off` and
`prolongation` appear far less often than `filled_pause`, so a sample chosen by
minutes will contain plenty of the latter and almost none of the former.
Select material until each P0 speech class has enough instances for its
per-class figure to mean something, and record how many that turned out to be.

**Several speakers, not one.** A single speaker's disfluency profile is
idiosyncratic — some people never produce `prolongation` at all — and agreement
measured on one speaker is agreement about that speaker.

**Blind.** Separate files, no shared tiers, no discussion until both are
finished. A second annotator who can see the first is reviewing, not
annotating, and the resulting figure is not agreement.

## What gets measured, and why in that order

`corpus agreement` reports three stages, because they are three separate
questions with three different fixes.

### 1. Did they find the same events?

Positive specific agreement over matched events, plus what each found alone.
Never counts a true negative, so the empty timeline cannot inflate it.

> **The trap this avoids.** Discretise the recording into 10 ms frames, label
> each with the class covering it or "none", compute Cohen's kappa: about 95%
> of frames are "none", both annotators agree on essentially all of them, and
> kappa comes back near 0.9 while telling you nothing. The coefficient measures
> the silence. `measures.kappa_over_time_frames` exists and raises, so the
> mistake has somewhere to fail loudly.

### 2. Did they draw the same boundaries?

Median and p95 start and end error over matched pairs.

**This is the human ceiling for NFR-004.** The engine claims a 250 ms boundary
tolerance and nobody has measured whether that is generous or optimistic. If
two trained annotators agree to 40 ms, a model claiming 250 has slack to
justify. If they disagree by 300, the target needs revisiting before anything
is trained.

#### The 250 ms is this project's own invention, and the literature is not encouraging

Worth stating plainly before two people are trained on it, because the honest
answer changes what they are asked to do.

**No disfluency annotation standard has a temporal dimension.** The Switchboard
disfluency stylebook — the canonical scheme — marks *spans of words*, never
points in time. The only Spanish disfluency corpus that confronted the question
(Rodríguez, Torres & Varona, DiSS'01) explicitly declined to assign duration to
pauses, filled pauses and lengthenings, and left alignment to the algorithms.
FluencyBank's time-aligned annotations were subsequently found unreliable and
discarded. A 250 ms boundary criterion is therefore not inherited from anywhere;
it is this project's addition.

**And the strictly easier task already scores badly.** Apple's SEP-28k measured
at least three trained annotators on *binary present/absent labels over pre-cut
three-second clips* — no timing component at all. Verbatim from §2.3 of the
paper:

> We measured Fleiss Kappa inter-annotator agreement and found word repetitions,
> interjections, sound repetitions, and no dysfluencies were more consistent
> (0.62, 0.57, 0.40, 0.39) and blocks and prolongations had only fair or slight
> agreement (0.25, 0.11).

Prolongation at **κ = 0.11** on a task with no clock in it. Asking two
annotators to place its onset and offset within 250 ms is asking for something
harder than the task that produced that number.

#### What the pilot does about it

- **Tight temporal agreement is scoped, not uniform.** Boundary error is
  reported per class, and the 250 ms review threshold is meaningful for the
  classes with a sharp acoustic edge — `cut_off`, `filled_pause`, `repetition`,
  `false_start`. `corpus agreement --boundary-review-ms` exists precisely so the
  threshold is an argument rather than a constant.
- **`prolongation` and `silent_pause` are reported presence-first.** A
  prolongation has no edge to agree on: it is a gradual lengthening, and the
  question "when did it start" has no answer the speaker's articulation
  supplies. Report specific agreement on detection, and boundary error at a
  wider window, stated.
- **Nothing here is decided in advance of the data.** Pilot B measures the
  ceiling per class. If `prolongation` comes back near the literature's 0.11,
  that is the finding, and NFR-004 gets scoped to the classes that support it
  rather than quietly averaged into a single number that survives review by
  being vague.

**Report the low coefficients.** A thesis that reports prolongation agreement
honestly and narrows its claims accordingly survives examination; one that
reports a single pooled κ, or none, does not.

### 3. Did they give it the same label?

Cohen's kappa and nominal Krippendorff's alpha over matched events only, the
per-class confusion matrix, and specific agreement for the classes §17 predicts
will be hardest: `false_start`, `self_repair`, and the contextual roles.

Contextual roles are measured **separately from classes**. Two annotators can
agree that "este" is a `lexical_filler` and disagree about whether it is a
`filler` or `semantic` — folding that into the class figure would hide a
problem with the role definitions behind a class that looks agreed.

### On Krippendorff's alpha for unitizing

Nominal alpha over matched events is implemented. Alpha-**u**, the version that
measures whether two annotators segmented a continuum the same way without
assuming shared units, is not — its difference function is intricate, there is
no widely-trusted Python implementation to check against, and a subtly wrong
alpha-u in a thesis is worse than an absent one.

Stages 1 and 2 address the same **question** — did they segment the timeline
the same way — in numbers whose computation is readable in
`corpus/agreement/measures.py`. They are **not the same measurement**, and the
thesis must not claim they are. Two differences a reviewer will find:

- **They are not chance-corrected.** Alpha-u is. Specific agreement and a
  boundary median are raw observed agreement, so two annotators marking events
  at random on a densely annotated recording show some agreement here and none
  under alpha-u.
- **They are two numbers on no common scale, not one coefficient.** Alpha-u
  gives a single value over the whole continuum, empty stretches included,
  where 0 is chance and 1 is perfect. The pair here deliberately never touches
  the empty timeline — which is why it cannot be inflated by it — and has no
  such scale: 0.78 specific agreement is the F1 between two annotators _at one
  matching threshold_, and it moves when the threshold moves. That is why every
  report prints the matching rule beside the number.

If a reviewer asks for alpha-u specifically, or the thesis needs a
chance-corrected unitizing coefficient, the two honest routes are to integrate
an established implementation and cite it, or to have the methodologist specify
the difference function and implement it against worked examples from
Krippendorff's own papers. See `corpus/agreement/report.py`.

## The same matching rule is used to score the model

The matching in `corpus.agreement.matching` is the one the model will be scored
with against the held-out set. That is deliberate and it is the most useful
property here: it puts the human ceiling and the model's F1 on one scale.

If two trained annotators reach 0.78 positive specific agreement on
`false_start`, then NFR-002's macro-F1 target of 0.80 has to be read against
0.78 rather than against 1.0 — and a model reporting 0.80 on that class is at
the ceiling, not below the target.

## Adjudication

After both annotators finish and the report is produced:

1. Walk every disagreement together.
2. Record each in the disagreement log (`DISAGREEMENT_LOG.md`) with the audio
   position, both readings, and what was decided.
3. Where the disagreement is a manual problem rather than a judgement call,
   amend the definition in `domain/shared/taxonomy.py`, bump
   `TAXONOMY_VERSION`, and regenerate the manual.
4. Produce the adjudicated annotation (`annotation_pass: adjudicated`). It is
   excluded from agreement computation by construction — including it would
   measure the resolution process rather than the annotators.

**A taxonomy change requires re-running Pilot B.** Amending a definition and
keeping the agreement figure measured under the old one reports a number for a
manual that no longer exists.

This is enforced, not just written down: `compare` refuses two files whose
`taxonomy_version` differs **at all** — major, minor or patch — and refuses a
file that records no version. An earlier version tolerated a minor difference
and printed a note, which contradicted this paragraph; the additive-class
argument for tolerating it does not cover a class whose *meaning* moved in a
minor release, and a note at the bottom of a report does not survive being
copied into a results table.

## Exit criterion

**What closes here is "Phase 0 — speech taxonomy", not Phase 0.** The scope
section above excludes the nine visual classes, so a pilot that never showed an
annotator a video frame cannot close a multimodal phase. Closing "Phase 0" on
the strength of this would be recording, in the project's own tracker, that
something was validated which was not looked at.

### Phase 0 — speech taxonomy

Closes when all four hold:

- [ ] Two annotators have independently annotated Pilot B.
- [ ] The agreement report is produced, committed, and its numbers are stable
      enough to interpret (the report flags samples too small for kappa).
- [ ] Every disagreement is in the log, resolved or explicitly marked
      unresolved with the reason.
- [ ] The manual reflects whatever the disagreements demanded, at its resulting
      version.

Only then does corpus recruitment start. Recruitment is an audio activity, so
it is not blocked by the visual half remaining open.

### Phase 0 — visual taxonomy

**Open. Nothing below has been attempted.**

- [ ] Decide whether the visual classes are events at all. `insufficient_lighting`
      is a continuous condition; two annotators marking it over overlapping
      stretches are unitizing, not detecting, and the matching in
      `corpus.agreement.matching` is the wrong instrument for that.
- [ ] Extend the annotation record, or define a second one, with whatever a
      visual observation actually needs — it has no words tier to check
      against, no contextual role and no lexical expression.
- [ ] Write the agreement conventions that follow from that decision.
- [ ] Run a visual pilot with its own protocol.

Scheduled for Phase 5, when the corpus exists and a taxonomy error costs
re-annotating one modality rather than re-recording everything.

**Any claim that "Phase 0 is closed" must name which half.** The two are
tracked separately here precisely so the sentence cannot be written without
one.

## Commands

```bash
# One template per annotator per recording. Pseudonyms only.
corpus template pilot-001-ana.eaf \
  --recording-id pilot-001 --speaker P-001 --annotator ana \
  --media pilot-001.wav

corpus template pilot-001-beto.eaf \
  --recording-id pilot-001 --speaker P-001 --annotator beto \
  --media pilot-001.wav

# After annotation.
corpus validate pilot-001-ana.eaf pilot-001-beto.eaf
corpus agreement pilot-001-ana.eaf pilot-001-beto.eaf
corpus agreement pilot-001-ana.eaf pilot-001-beto.eaf --json > agreement-pilot-001.json
```
