"""Opt-in live smoke test for the grounded DeepSeek diagnosis path."""

from __future__ import annotations

import os

import pytest

from app.kg.loader import load_seed_graph
from app.kg.retrieval import retrieve
from app.llm.client import DeepSeekClient
from app.pipeline.diagnose import DiagnosisStatus, diagnose

pytestmark = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not configured; live smoke test skipped",
)


def test_live_deepseek_hit_is_grounded_to_retrieved_subgraph() -> None:
    query = "叶片出现褐色病斑"

    with load_seed_graph() as graph:
        retrieval_result = retrieve(graph, query)
        assert retrieval_result.is_hit

        allowed_entity_ids = {
            node.entity_id for node in retrieval_result.subgraph.nodes
        }
        llm = DeepSeekClient()
        try:
            result = diagnose(graph, query, llm)
        finally:
            llm.close()

    assert result.status is DiagnosisStatus.DIAGNOSED
    assert result.diagnosis is not None
    assert result.diagnosis.reference_ids
    assert set(result.diagnosis.reference_ids) <= allowed_entity_ids

    for suggestion in result.model_suggestions:
        assert suggestion.is_model_suggestion is True
        assert suggestion.reference_ids
        assert set(suggestion.reference_ids) <= allowed_entity_ids
