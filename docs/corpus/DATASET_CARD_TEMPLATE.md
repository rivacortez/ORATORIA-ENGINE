# Dataset card — template

**Status:** template only. **No card has been filled, because no recording
session has happened.** Blocker: no recorded corpus — `docs/BUILD_RECORD.md` §5.

**Phase 1 deliverable.** `BASELINES.md` §4 requires that dataset cards record
provenance, consent basis, recording conditions and partition checksums. One of
those four is produced by `corpus split --freeze`; the other three are prose a
human writes, and this is the shape of that prose.

---

## Why the record exists before the values

`BASELINES.md` §4 states the rule and names no corpus.
`src/corpus/partition/freeze.py` produces the machine-readable half and says in
its own docstring that the half a human writes "is prose and belongs beside it,
not in it" — and until this file existed there was nowhere beside it to put it.
A rule with no instance is a rule nobody has had to follow yet, and the first
person to record a speaker will record them without writing any of this down.

## What every field below has in common

**It is cheap at recruitment and impossible afterwards.** A speaker who has
finished their session and gone is a speaker nobody can ask which language they
grew up speaking, whether they were given the topic in advance, or whether they
agreed to their audio being used outside this thesis. The recordings keep the
sound. They keep nothing about how the sound came to exist, and the annotation
files keep less: they are pseudonymous by construction, so there is no route
from a datum back to the person who could answer.

This is why the card is filled **as the sessions happen**. A card written after
the last session is a card written from memory, and a corpus documented from
memory is one a reviewer is entitled to distrust.

## How to fill it

- One card per frozen corpus, committed beside the manifest that
  `corpus split --freeze` writes.
- A field that cannot be filled yet reads `_pending — <what blocks it>_`. Never
  blank and never invented: a blank cell reads as "nothing to say about this",
  and an invented one cannot be falsified by anyone who was not in the room.
  `REFERENCE_ENVIRONMENT.md` and `BASELINE_PINS.md` are filled that way and are
  the register to match.
- Values the manifest already supplies are **not** copied into this card. See §1.
- Some of these facts may later become structured fields on the annotation
  record. When they do, the card cites the record rather than restating it.
  Until then the card is the only place they exist at all.

---

## 1. Supplied by the freeze manifest — do not retype

`corpus split --freeze` writes these, and `corpus verify` checks them months
later. They are listed here so a reader knows the card is not silent about
them, and so nobody hand-copies a digest.

| Manifest key | What it establishes | Why it is not a card field |
| --- | --- | --- |
| `digest` | The identity of the whole assignment — which corpus a number was computed over. | A digest typed by hand is a second source that can disagree with the first, and only one of the two is the one `corpus verify` reads. |
| `sha256` | Per annotation file, whether its bytes changed since the freeze. | Sixty-four hex characters transcribed by a human is a transcription error waiting to be believed. |
| `partition` | Which of train, dev and held-out each recording is in. | Restating the split in prose creates a version of it that no tool checks, and a stale copy of a partition table is how a held-out recording ends up in training. |
| `recording_id` and `source_path` | Which file each entry refers to. | Mechanically derived from the files that were frozen; retyping adds no fact and one more place to be wrong. |
| `speaker_pseudonym` | Which recordings belong to the same speaker. | The card describes the population; the manifest holds the per-speaker mapping, and duplicating pseudonyms into prose invites someone to annotate one with a real name. |
| `duration_ms` | How much audio each recording contributes. | Summing durations by hand produces a corpus size nobody can reproduce. |
| `class_counts` | How many instances of each class each recording carries. | This is the number `corpus inventory` and the adequacy check read; a prose total drifts from it the first time a file is re-exported. |
| `seed` and `shares` | How the split was drawn, so it can be drawn again. | The split is reproducible from these two values; a prose description of the procedure is not. |
| `schema_version` and `taxonomy_version` | Which record shape and which manual the corpus was annotated under. | The freeze refuses a corpus that disagrees with itself about either, so the manifest's value is the checked one. |
| `manifest_version` | Which manifest shape the file was written in. | Read by the loader to refuse a manifest it would otherwise misinterpret; a human copy has no reader. |

**What the manifest does not answer.** Everything in §2 to §5. It records what
was frozen, never who was recorded, on what basis, or under what conditions.

---

## 2. Provenance

Who the speakers are as a population, and how they came to be in the corpus.

| Field | Value | Why it is gone if it is not captured now |
| --- | --- | --- |
| Population the speakers were drawn from — institution, programme, level | _pending — no recruitment has started_ | A pseudonym carries no institution, and the people who could say which one have moved on by the time anyone asks. |
| Recruitment channel — how each speaker was reached | _pending — no recruitment has started_ | Self-selection is a property of the channel, not of the speaker, and nothing in a recording says which channel produced it. |
| Number approached and number who recorded | _pending — no recruitment has started_ | The refusal rate is the only measure of self-selection bias, and only the person recruiting ever sees the denominator. |
| Inclusion and exclusion criteria actually applied | _pending — no recruitment has started_ | A criterion applied in someone's head leaves no trace: the corpus shows who is in it, never who was turned away or why. |
| Recruitment window — first and last session date | _pending — no recruitment has started_ | File timestamps survive only until the files are copied, and a copy resets them. |
| Compensation, course credit or none | _pending — no recruitment has started_ | It changes who volunteers, and no part of the audio records it. |
| Speaking task — the prompt, the topic, and whether it was given in advance | _pending — no recording session has happened_ | Disfluency rates depend on the task more than on the speaker; unless the prompt was spoken aloud it is not in the recording at all. |
| Register — spontaneous, semi-prepared or read aloud | _pending — no recording session has happened_ | Sometimes audible, never reliably, and never recoverable from an annotation file. |
| Audience present, and how many | _pending — no recording session has happened_ | It changes delivery and leaves no trace in a single-speaker recording. |
| Regional variety, and whether the speaker stated it or somebody assigned it | _pending — no recruitment has started_ | A variety code does not say who decided it, and a self-reported variety and an assigned one support different claims. |
| Language background — first language, other languages, coarse only | _pending — no recruitment has started_ | Only the speaker knows it, and after pseudonymisation there is no route back to ask. |
| Coarse demographics collected, and which were deliberately not collected | _pending — no recruitment has started_ | §14.2 asks for error analysis by subgroup; a subgroup nobody recorded cannot be analysed, and "we chose not to ask" and "we forgot" are indistinguishable later. |
| Who assigned the pseudonyms, and whether a re-identification key exists | _pending — no recruitment has started_ | If a key exists its custody is a consent question; if it does not, a withdrawal request cannot be honoured — and afterwards nobody can tell which of the two is true. |

---

## 3. Consent basis

What each speaker agreed to, in which words, and where the evidence is.
`CONSENT_AND_RETENTION.md` is the policy. This section records the instance.

| Field | Value | Why it is gone if it is not captured now |
| --- | --- | --- |
| Consent policy version in force at the session | _pending — no recording session has happened_ | The policy is versioned precisely because its wording changes; a consent record naming no version attests to text nobody can reproduce. |
| Participant-facing text version actually read to the speaker | _pending — no recording session has happened_ | §6 of the policy is the only text the speaker ever saw, and what they agreed to is that text, not the policy summarising it. |
| Consent modality — signed paper, digital form, recorded verbal — and where it is held | _pending — no recording session has happened_ | After the fact, "they agreed" is an assertion rather than a basis, and the artifact that would settle it is the one nobody made. |
| Audio consent, per speaker | _pending — no recording session has happened_ | Not derivable from the existence of an audio file: the file exists either way. |
| Video consent, recorded separately from audio | _pending — no recording session has happened_ | The taxonomy has a visual half that Phase 5 will annotate. A corpus where video consent was assumed from audio consent cannot be used for it, and which speakers granted which is not recoverable from the recordings. |
| Retention decision — ephemeral, or an explicit policy with a positive TTL | _pending — no recording session has happened_ | The default is deletion. Raw media kept without a stated TTL was kept without a basis, and that fact disappears the moment the file is copied somewhere else. |
| Use outside this thesis — publication, redistribution, secondary research | _pending — no recording session has happened_ | This is the field that decides whether the corpus can ever be released. It can only be granted by the speaker, at recruitment, and never inferred afterwards. |
| Third parties audible or visible in the recording, and their basis | _pending — no recording session has happened_ | A bystander who was never identified cannot be found later, and their recording cannot be justified retrospectively. |
| Ethics approval — committee, protocol number, decision date | _pending — nothing in this repository records an approval; obtain or record the reference before recruitment_ | An approval obtained after the recordings does not cover them, and a reference reconstructed from memory is not a reference. |
| Withdrawal route that survives pseudonymisation | _pending — no recruitment has started_ | A speaker who cannot be matched to their recordings cannot have them deleted, so the promise of withdrawal is only as real as the route recorded here. |
| What a withdrawal does to an already-frozen partition | _pending — decide before the first freeze_ | Removing a recording changes the digest, so the answer determines whether earlier numbers still refer to anything. Deciding it under pressure, after a withdrawal, is deciding it badly. |

---

## 4. Recording conditions

`REFERENCE_ENVIRONMENT.md` records the *machines*. This section records the
*sessions*, which may differ per speaker and are not a property of any machine.

| Field | Value | Why it is gone if it is not captured now |
| --- | --- | --- |
| Room, and approximate size | _pending — no recording session has happened_ | Reverberation is audible but not measurable from the recording alone, and the room may not exist in the same state a year later. |
| Ambient noise floor, measured | _pending — no recording session has happened_ | Measurable at the time in seconds; afterwards inseparable from the speaker's own signal. |
| Microphone model | _pending — no recording session has happened_ | Not stored in a WAV file. Two microphones with different frequency responses produce corpora that are not comparable, and nothing in the bytes says which was used. |
| Microphone placement and distance from the speaker | _pending — no recording session has happened_ | It sets the level and the proximity effect, and it is a fact about the room rather than about the file. |
| Physical capture device confirmed from the recorded file | _pending — no recording session has happened_ | A settings dialog reports what was requested; the file reports what happened. The check takes a minute at the session and is impossible once the session is over. |
| Sample rate, bit depth and channel count as captured | _pending — no recording session has happened_ | Readable from the file, but only says what the file holds: audio captured at 8 kHz and resampled to 16 kHz reads as 16 kHz forever. |
| Container and codec, and any lossy step in the chain | _pending — no recording session has happened_ | A lossy encode removes exactly the low-energy detail that `cut_off` and `prolongation` boundaries depend on, and the result does not announce itself. |
| Enhancement or suppression in the path, and confirmation that it was bypassed | _pending — no recording session has happened_ | Noise removal is built to suppress breath, creak and glottal stops — the acoustic evidence for three P0 classes. Enhanced audio sounds clean, so nothing later reveals that the corpus encodes the enhancer's decisions as ground truth. |
| Recording application and version | _pending — no recording session has happened_ | It decides the default gain, the resampler and the container; the file records none of them. |
| Session date and time of day | _pending — no recording session has happened_ | Recoverable from file metadata only until the first copy, and time of day is a plausible source of voice variation. |
| Camera model, resolution and frame rate | _pending — Phase 5; the Phase 0.5 pilots show no video_ | Needed before any visual class can be annotated, and a camera swapped between sessions is invisible in the frames. |
| Camera position relative to eye level | _pending — Phase 5; the Phase 0.5 pilots show no video_ | Gaze is estimated against the speaker's own calibration, and the geometry that calibration corrects for is a fact about the room, not about the video. |
| Lighting — source, direction, approximate level | _pending — Phase 5; needs a light meter_ | `insufficient_lighting` is a published class. Judging it from the footage afterwards is judging the class by the evidence it is supposed to explain. |
| Anything that changed part-way through the corpus | _pending — no recording session has happened_ | The field people forget. A microphone or room changed at speaker twelve makes two corpora, and nobody notices unless somebody wrote it down at the time. |

---

## 5. Annotation provenance

Who produced the labels, under which manual, and how much agreement there is.

| Field | Value | Why it is gone if it is not captured now |
| --- | --- | --- |
| Annotators, by pseudonymous id | _pending — no annotator has been recruited_ | The manifest records which files exist, never who made them or how many people were involved. |
| Training each annotator received before the first pass | _pending — no annotator has been recruited_ | Agreement between two people trained together and two trained apart means different things, and the difference is not in the numbers. |
| Manual version each annotator worked under | _pending — no annotation exists_ | The manifest carries one taxonomy version for the corpus. It cannot express a corpus annotated across a change, which the freeze refuses precisely because it would be unreadable. |
| Which recordings received two independent passes | _pending — no annotation exists_ | Agreement can only be reported for these. A corpus that mixes single and double annotation without saying so reports a figure covering part of itself as if it covered all of it. |
| Adjudication policy, and where the log is | _pending — no annotation exists_ | `DISAGREEMENT_LOG.md` records the arguments; the policy that governed them is a decision made once and otherwise never written down. |
| Agreement figures achieved, and the report they came from | _pending — Pilot B has not been run_ | A number without its report is a number nobody can regenerate; §14.2 needs the report, not the summary. |
| Classes the annotators could not converge on | _pending — Pilot B has not been run_ | These are findings about the taxonomy, and §14.4 requires them to be preserved rather than resolved away. |

---

## 6. Known gaps

Filled in as they are discovered, not at the end. A limitation found late and
recorded late is indistinguishable from one that was hidden.

**None recorded — there is no corpus to have gaps.** The table stays empty
visibly, in the shape it gets filled in: a class below the adequacy threshold,
a speaker who withdrew after the freeze, a session recorded under conditions
that turned out not to match the rest.

| Gap | Status | What it blocks |
| --- | --- | --- |

---

## 7. Card metadata

| Field | Value | Why it is here |
| --- | --- | --- |
| Card version | _pending — no card exists_ | The card is versioned with the corpus it describes, so a citation resolves to the text that was true then. |
| Corpus manifest this card accompanies | _pending — no manifest has been written_ | A card that names no manifest describes an unspecified corpus. |
| Filled on | _pending — no card exists_ | The gap between the sessions and the writing is the measure of how much came from memory. |
| Filled by | _pending — no card exists_ | Someone has to be answerable for each claim above. |
| Verified against | _pending — no card exists_ | Which claims were checked against a file or a receipt, and which rest on the writer's word. |
