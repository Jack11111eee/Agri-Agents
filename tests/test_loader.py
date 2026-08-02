from __future__ import annotations

import json

import pytest

from app.kg.loader import DEFAULT_SEED_PATH, load_seed_graph


@pytest.fixture
def graph():
    store = load_seed_graph()
    yield store
    store.close()


def _scalar(connection, query: str):
    return connection.execute(query).get_all()[0][0]


def test_load_seed_graph_creates_schema_and_four_rice_entities(graph):
    connection = graph.connection

    assert _scalar(connection, "MATCH (d:Disease) RETURN count(d)") == 4
    assert _scalar(connection, "MATCH (s:Symptom) RETURN count(s)") == 8
    assert _scalar(connection, "MATCH (m:ControlMethod) RETURN count(m)") == 4
    assert _scalar(connection, "MATCH (p:Pesticide) RETURN count(p)") == 4
    assert _scalar(connection, "MATCH ()-[r:HAS_SYMPTOM]->() RETURN count(r)") == 8
    assert _scalar(connection, "MATCH ()-[r:HAS_CONTROL]->() RETURN count(r)") == 4
    assert _scalar(connection, "MATCH ()-[r:RECOMMENDS_PESTICIDE]->() RETURN count(r)") == 4


def test_every_loaded_relationship_contains_provenance(graph):
    rows = graph.connection.execute(
        """
        MATCH ()-[r]->()
        RETURN r.source, r.version, r.confidence
        """
    ).get_all()

    assert len(rows) == 16
    assert all(source and version and 0 <= confidence <= 1 for source, version, confidence in rows)


def test_seed_data_is_valid_json_and_does_not_contain_credentials():
    payload = json.loads(DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False).lower()

    assert payload["crop"] == "水稻"
    assert 3 <= len(payload["diseases"]) <= 5
    assert "api_key" not in serialized
    assert "secret" not in serialized


def test_missing_seed_file_bubbles_a_clear_filesystem_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_seed_graph(seed_path=tmp_path / "missing.json")


def test_seed_relation_without_provenance_is_rejected_before_loading(tmp_path):
    payload = json.loads(DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
    del payload["diseases"][0]["symptoms"][0]["source"]
    invalid_seed = tmp_path / "invalid-seed.json"
    invalid_seed.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source, version, and confidence"):
        load_seed_graph(seed_path=invalid_seed)
