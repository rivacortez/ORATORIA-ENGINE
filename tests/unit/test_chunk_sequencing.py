"""FR-008: missing, duplicated and out-of-order chunks are detected as such.

The subtle requirement is §6.3's: a gap must be detected *through* `chunk_seq`,
not inferred from silence. A dropped packet and a speaker pausing look
identical in the waveform, and only the sequence numbers separate them. Getting
this wrong makes the engine report a silent pause the speaker never took -
which then becomes evidence in a report.

The second subtlety is that reordering is normal on an unreliable transport, so
loss is only confirmed at close. Treating the first out-of-order arrival as
loss would fill a healthy session with false quality warnings.
"""

from __future__ import annotations

import pytest

from evidence_engine.domain.sessions.sequencing import ChunkLedger, ChunkVerdict, SequenceGap


def test_chunks_in_order_are_accepted() -> None:
    ledger = ChunkLedger()

    verdicts = [ledger.offer(n) for n in range(5)]

    assert verdicts == [ChunkVerdict.ACCEPTED] * 5
    assert ledger.finalize() == ()


def test_a_repeated_chunk_is_a_duplicate_not_an_error() -> None:
    """§7.4 makes redelivery idempotent."""
    ledger = ChunkLedger()
    ledger.offer(0)
    ledger.offer(1)

    assert ledger.offer(1) is ChunkVerdict.DUPLICATE
    assert ledger.duplicates == 1
    assert ledger.finalize() == ()


def test_a_forward_jump_signals_a_gap() -> None:
    ledger = ChunkLedger()
    ledger.offer(0)

    assert ledger.offer(3) is ChunkVerdict.GAP_DETECTED


def test_reordering_is_not_reported_as_loss() -> None:
    """Chunks 1 and 2 arrive late; nothing was actually lost."""
    ledger = ChunkLedger()
    ledger.offer(0)
    ledger.offer(3)
    ledger.offer(2)
    ledger.offer(1)

    assert ledger.finalize() == ()
    assert ledger.missing_count == 0


def test_genuine_loss_is_confirmed_at_close() -> None:
    ledger = ChunkLedger()
    for n in (0, 1, 5, 6):
        ledger.offer(n)

    gaps = ledger.finalize()

    assert gaps == (SequenceGap(2, 4),)
    assert ledger.missing_count == 3


def test_several_separate_gaps_are_kept_separate() -> None:
    """FR-024 scopes availability per window, so the ledger reports where."""
    ledger = ChunkLedger()
    for n in (0, 2, 3, 7):
        ledger.offer(n)

    assert ledger.finalize() == (SequenceGap(1, 1), SequenceGap(4, 6))


def test_a_late_chunk_after_a_gap_is_marked_out_of_order() -> None:
    ledger = ChunkLedger()
    ledger.offer(0)
    ledger.offer(5)

    assert ledger.offer(2) is ChunkVerdict.OUT_OF_ORDER


def test_a_negative_sequence_number_is_rejected() -> None:
    ledger = ChunkLedger()

    with pytest.raises(ValueError, match="non-negative"):
        ledger.offer(-1)


def test_a_gap_reports_how_many_chunks_it_covers() -> None:
    assert SequenceGap(4, 4).count == 1
    assert SequenceGap(2, 6).count == 5
