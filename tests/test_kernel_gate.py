from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.kg.loader import load_seed_graph
from app.kg.retrieval import RETRIEVAL_CYPHER, RetrievalStatus, retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "scripts" / "query_kg.py"


@pytest.fixture
def graph():
    store = load_seed_graph()
    yield store
    store.close()


def test_kernel_gate_returns_stable_subgraph_with_provenance(graph):
    result = retrieve(graph.connection, "叶片出现褐色病斑")

    assert result.status is RetrievalStatus.HIT
    assert result.matched_symptoms == ["叶片出现褐色病斑"]
    assert {(node.node_type, node.node_id) for node in result.subgraph.nodes} == {
        ("Disease", "disease:rice-blast"),
        ("Symptom", "symptom:rice-blast-brown-lesions"),
        ("ControlMethod", "control:rice-blast-resistant-variety"),
        ("Pesticide", "pesticide:tricyclazole"),
    }
    assert {
        (relationship.relation_type, relationship.source_id, relationship.target_id)
        for relationship in result.subgraph.relationships
    } == {
        ("HAS_SYMPTOM", "disease:rice-blast", "symptom:rice-blast-brown-lesions"),
        ("HAS_CONTROL", "disease:rice-blast", "control:rice-blast-resistant-variety"),
        (
            "RECOMMENDS_PESTICIDE",
            "control:rice-blast-resistant-variety",
            "pesticide:tricyclazole",
        ),
    }
    assert len(result.subgraph.nodes) == 4
    assert len(result.subgraph.relationships) == 3
    for relationship in result.subgraph.relationships:
        assert set(relationship.properties) == {"source", "version", "confidence"}
        assert relationship.properties["source"]
        assert relationship.properties["version"]
        assert 0 <= relationship.properties["confidence"] <= 1


def test_kernel_gate_fixes_minimum_match_threshold_and_abstention(graph):
    two_symptom_hit = retrieve(
        graph.connection,
        ["叶片出现褐色病斑", "叶片出现梭形病斑"],
        min_matched_symptoms=2,
    )
    assert two_symptom_hit.status is RetrievalStatus.HIT
    assert two_symptom_hit.matched_symptoms == ["叶片出现褐色病斑", "叶片出现梭形病斑"]

    threshold_abstention = retrieve(
        graph.connection,
        "叶片出现褐色病斑",
        min_matched_symptoms=2,
    )
    assert threshold_abstention.status is RetrievalStatus.ABSTAIN
    assert threshold_abstention.subgraph == type(threshold_abstention.subgraph)()
    assert "匹配症状数不足" in threshold_abstention.reason

    unknown_abstention = retrieve(graph.connection, "不存在的症状XYZ")
    assert unknown_abstention.status is RetrievalStatus.ABSTAIN
    assert unknown_abstention.subgraph == type(unknown_abstention.subgraph)()
    assert "未找到" in unknown_abstention.reason


def test_kernel_gate_records_parameterized_query_for_untrusted_input(graph):
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
    assert "$symptom_feature" in RETRIEVAL_CYPHER
    assert symptom not in RETRIEVAL_CYPHER


def test_kernel_gate_hit_and_abstention_never_import_llm_clients(graph, monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"openai", "anthropic"}:
            raise AssertionError(f"LLM client import attempted: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert retrieve(graph.connection, "叶片出现褐色病斑").status is RetrievalStatus.HIT
    assert retrieve(graph.connection, "不存在的症状XYZ").status is RetrievalStatus.ABSTAIN
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules


@pytest.mark.parametrize(
    ("symptom", "expected_status", "expected_reason"),
    [
        ("叶片出现褐色病斑", "HIT", "命中"),
        ("不存在的症状XYZ", "ABSTAIN", "未找到"),
    ],
)
def test_kernel_gate_cli_prints_observable_json(symptom, expected_status, expected_reason):
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--symptoms", symptom, "--json"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == expected_status
    assert expected_reason in payload["reason"]
    if expected_status == "HIT":
        assert payload["subgraph"]["nodes"]
        assert payload["subgraph"]["relationships"]
        assert all(
            set(edge["properties"]) == {"source", "version", "confidence"}
            for edge in payload["subgraph"]["relationships"]
        )
    else:
        assert payload["subgraph"] == {"nodes": [], "relationships": []}
