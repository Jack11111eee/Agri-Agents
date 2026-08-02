from __future__ import annotations

import sys

import pytest

from app.kg.loader import load_seed_graph
from app.kg.retrieval import RETRIEVAL_CYPHER, RetrievalStatus, retrieve


@pytest.fixture
def graph():
    store = load_seed_graph()
    yield store
    store.close()


def test_known_symptom_returns_structured_provenance_subgraph(graph):
    result = retrieve(graph.connection, "叶片出现褐色病斑")

    assert result.status is RetrievalStatus.HIT
    assert result.matched_symptoms == ["叶片出现褐色病斑"]
    assert {node.node_type for node in result.subgraph.nodes} == {
        "Disease",
        "Symptom",
        "ControlMethod",
        "Pesticide",
    }
    assert len(result.subgraph.relationships) == 3
    for relationship in result.subgraph.relationships:
        assert relationship.properties["source"]
        assert relationship.properties["version"]
        assert 0 <= relationship.properties["confidence"] <= 1


def test_unknown_symptom_returns_explicit_abstention(graph):
    result = retrieve(graph.connection, "不存在的症状XYZ")

    assert result.status is RetrievalStatus.ABSTAIN
    assert result.subgraph.nodes == []
    assert result.subgraph.relationships == []
    assert "未找到" in result.reason


def test_threshold_failure_abstains_even_when_a_candidate_exists(graph):
    result = retrieve(graph.connection, "叶片出现褐色病斑", min_matched_symptoms=2)

    assert result.status is RetrievalStatus.ABSTAIN
    assert result.subgraph.nodes == []
    assert "不足" in result.reason


def test_empty_and_invalid_inputs_are_explicit(graph):
    empty_result = retrieve(graph.connection, "   ")

    assert empty_result.status is RetrievalStatus.ABSTAIN
    assert "未提供" in empty_result.reason
    with pytest.raises(ValueError, match="at least 1"):
        retrieve(graph.connection, "叶片出现褐色病斑", min_matched_symptoms=0)
    with pytest.raises(TypeError, match="each symptom feature"):
        retrieve(graph.connection, ["叶片出现褐色病斑", 3])


def test_user_input_is_bound_as_a_cypher_parameter(graph):
    class RecordingConnection:
        def __init__(self, connection):
            self.connection = connection
            self.calls = []

        def execute(self, query, parameters=None):
            self.calls.append((query, parameters))
            return self.connection.execute(query, parameters)

    connection = RecordingConnection(graph.connection)
    symptom = "不存在') MATCH (n) DETACH DELETE n //"

    result = retrieve(connection, symptom)

    assert result.status is RetrievalStatus.ABSTAIN
    assert connection.calls == [(RETRIEVAL_CYPHER, {"symptom_feature": symptom})]
    assert symptom not in RETRIEVAL_CYPHER


def test_retrieval_import_does_not_load_llm_clients():
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules
