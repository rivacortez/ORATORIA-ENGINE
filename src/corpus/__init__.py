"""Corpus tooling: annotation schema, import/export and agreement measurement.

A sibling of ``evidence_engine``, not a part of it. The engine is a running
service; this is research tooling that produces the corpus the engine will
later be evaluated against. They share exactly one thing - the published
taxonomy - and an import-linter contract holds that boundary: ``corpus`` may
import ``evidence_engine.domain`` and nothing else from the engine.

That boundary is not tidiness. Sharing the taxonomy is the whole point: a
manual or an annotation schema that drifts from the allowlist produces
disagreement that looks like a hard boundary case and is really a
documentation bug. Sharing anything *else* would let corpus tooling depend on
a service's transaction handling, and the service is the thing that has to
stay deployable.
"""
