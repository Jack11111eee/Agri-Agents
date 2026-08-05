from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Sequence

import pytest

from app.kg.loader import load_seed_graph
from app.kg.retrieval import retrieve
from app.llm.protocol import ChatMessage
from app.pipeline.diagnose import (
    DiagnosisStatus,
    LLMClientRequiredError,
    LLMOutputError,
    diagnose,
)


@dataclass
class StubLLM:
    response: str = ""
    error: Exception | None = None
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.response


class ForbiddenLLM:
    def generate(self, messages: Sequence[ChatMessage]) -> str:
        raise AssertionError(f"LLM must not be called, received: {messages!r}")


@pytest.fixture
def graph():
    store = load_seed_graph()
    yield store
    store.close()


def llm_json(*, suggestions: list[dict[str, object]] | None = None) -> str:
    """Build a provider response under the ID-only contract (D006)."""

    return json.dumps(
        {
            "model_suggestions": suggestions
            if suggestions is not None
            else [
                {"referenced_entity_ids": ["control:rice-blast-resistant-variety"]}
            ],
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize("symptom_text", [None, 42, "", "  \n\t  "])
def test_invalid_input_returns_stable_empty_sections_without_llm(graph, symptom_text):
    result = diagnose(graph.connection, symptom_text, llm_client=ForbiddenLLM())

    assert result.status is DiagnosisStatus.INVALID_INPUT
    assert result.reason == "输入必须是非空文本"
    assert result.to_dict() == {
        "status": "INVALID_INPUT",
        "reason": "输入必须是非空文本",
        "matched_symptoms": [],
        "diagnosis": None,
        "verified_knowledge": [],
        "model_suggestions": [],
        "evidence_chain": [],
        "grounding_rejections": 0,
    }


def test_retrieval_abstention_short_circuits_llm_and_preserves_reason(graph):
    symptom_text = "不存在的症状XYZ"
    retrieval_result = retrieve(graph.connection, symptom_text)

    result = diagnose(graph.connection, symptom_text, llm_client=ForbiddenLLM())

    assert result.status is DiagnosisStatus.ABSTAINED
    assert result.reason == retrieval_result.reason
    assert result.matched_symptoms == retrieval_result.matched_symptoms
    assert result.diagnosis is None
    assert result.verified_knowledge == []
    assert result.model_suggestions == []
    assert result.evidence_chain == []


def test_hit_calls_llm_once_and_assembles_grounded_layers_from_subgraph(graph):
    symptom_text = "叶片出现褐色病斑"
    llm = StubLLM(response=llm_json())
    retrieval_result = retrieve(graph.connection, symptom_text)

    result = diagnose(graph.connection, symptom_text, llm_client=llm)

    assert result.status is DiagnosisStatus.DIAGNOSED
    assert len(llm.calls) == 1
    assert [message["role"] for message in llm.calls[0]] == ["system", "user"]
    assert "只能依据" in llm.calls[0][0]["content"]
    assert "不得返回 diagnosis、text" in llm.calls[0][0]["content"]
    assert "subgraph" in llm.calls[0][1]["content"]

    allowed_ids = {node.node_id for node in retrieval_result.subgraph.nodes}
    assert result.diagnosis is not None
    assert set(result.diagnosis.referenced_entity_ids) <= allowed_ids
    assert all(set(item.referenced_entity_ids) <= allowed_ids for item in result.model_suggestions)
    assert all(item.authoritative is False for item in result.model_suggestions)
    assert all(item.label == "模型补充建议" for item in result.model_suggestions)

    assert result.verified_knowledge == [
        node.to_dict() for node in retrieval_result.subgraph.nodes
    ]
    assert result.evidence_chain == [
        relationship.to_dict() for relationship in retrieval_result.subgraph.relationships
    ]
    for relationship in result.evidence_chain:
        assert set(relationship["properties"]) == {"source", "version", "confidence"}


def test_valid_entity_ids_cannot_launder_free_text_claims(graph):
    """HIGH-1: a real pesticide ID must not carry a fabricated dose/timing."""

    fabricated = "三环唑每亩 100 克，抽穗期喷施，安全间隔期 7 天。"
    llm = StubLLM(
        response=json.dumps(
            {
                "model_suggestions": [
                    {
                        "text": fabricated,
                        "referenced_entity_ids": ["pesticide:tricyclazole"],
                    }
                ]
            },
            ensure_ascii=False,
        )
    )

    with pytest.raises(LLMOutputError) as exc_info:
        diagnose(graph.connection, "叶片出现褐色病斑", llm_client=llm)

    # The rejection must not echo the prescription back to any caller or log.
    assert fabricated not in str(exc_info.value)


def test_provider_diagnosis_field_is_rejected_outright(graph):
    """HIGH-1: the provider has no authority to state a diagnosis."""

    unsafe_text = "图谱外病害乙是最终诊断。"
    llm = StubLLM(
        response=json.dumps(
            {
                "diagnosis": {
                    "text": unsafe_text,
                    "referenced_entity_ids": ["disease:rice-blast"],
                },
                "model_suggestions": [],
            },
            ensure_ascii=False,
        )
    )

    with pytest.raises(LLMOutputError) as exc_info:
        diagnose(graph.connection, "叶片出现褐色病斑", llm_client=llm)

    assert unsafe_text not in str(exc_info.value)


def test_grounding_filters_out_of_subgraph_selections_and_renders_graph_facts(graph):
    llm = StubLLM(
        response=llm_json(
            suggestions=[
                {"referenced_entity_ids": ["control:rice-blast-resistant-variety"]},
                {"referenced_entity_ids": ["pesticide:not-in-subgraph"]},
            ]
        )
    )

    result = diagnose(graph.connection, "叶片出现褐色病斑", llm_client=llm)

    assert len(result.model_suggestions) == 1
    assert "选用抗病品种并加强肥水管理" in result.model_suggestions[0].text
    assert result.grounding_rejections == 1
    assert "not-in-subgraph" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_pesticide_selection_renders_only_stored_safety_facts(graph):
    result = diagnose(
        graph.connection,
        "叶片出现褐色病斑",
        llm_client=StubLLM(
            response=llm_json(
                suggestions=[{"referenced_entity_ids": ["pesticide:tricyclazole"]}]
            )
        ),
    )

    text = result.model_suggestions[0].text
    assert "三环唑" in text
    assert "使用前必须核对当地登记标签、适用作物和安全间隔期" in text
    # No dose or timing exists in the graph, so none may appear in the output.
    assert "每亩" not in text
    assert "抽穗期" not in text


def test_diagnosis_is_deterministically_rendered_from_retrieved_disease(graph):
    result = diagnose(
        graph.connection,
        "叶片出现梭形病斑",
        llm_client=StubLLM(response=llm_json(suggestions=[])),
    )

    assert result.diagnosis is not None
    assert result.diagnosis.text == "图谱匹配到可能相关病害：稻瘟病"
    assert result.diagnosis.referenced_entity_ids == ("disease:rice-blast",)


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "{}",
        json.dumps({"model_suggestions": [{"referenced_entity_ids": []}]}),
        json.dumps({"model_suggestions": [{}]}),
        json.dumps({"model_suggestions": "not-a-list"}),
    ],
)
def test_malformed_llm_output_raises_explicit_error(graph, response):
    with pytest.raises(LLMOutputError):
        diagnose(
            graph.connection,
            "叶片出现褐色病斑",
            llm_client=StubLLM(response=response),
        )


def test_retrieval_failure_bubbles_before_llm(monkeypatch):
    retrieval_error = RuntimeError("graph connection lost")

    def fail_retrieval(connection, symptom_text):
        raise retrieval_error

    pipeline_module = sys.modules["app.pipeline.diagnose"]
    monkeypatch.setattr(pipeline_module, "retrieve", fail_retrieval)

    with pytest.raises(RuntimeError) as exc_info:
        diagnose(object(), "叶片出现褐色病斑", llm_client=ForbiddenLLM())

    assert exc_info.value is retrieval_error


def test_provider_failure_bubbles_without_returning_partial_model_content(graph):
    provider_error = TimeoutError("provider timed out")

    with pytest.raises(TimeoutError) as exc_info:
        diagnose(
            graph.connection,
            "叶片出现褐色病斑",
            llm_client=StubLLM(error=provider_error),
        )

    assert exc_info.value is provider_error


def test_hit_requires_an_injected_llm_client(graph):
    with pytest.raises(LLMClientRequiredError):
        diagnose(graph.connection, "叶片出现褐色病斑")


def test_pipeline_import_does_not_load_provider_sdk():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.pipeline; assert 'openai' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_connection_lock_is_released_before_the_llm_call(graph):
    """MEDIUM-2b: the retrieval lock must not be held across the network call.

    If the lock still wrapped the provider round trip, a single hit would block
    every other diagnosis for the LLM's multi-second timeout. We prove release
    by re-acquiring the same lock non-blocking from inside generate().
    """

    lock = threading.Lock()
    lock_free_during_generate: dict[str, bool] = {}

    class LockProbingLLM:
        def generate(self, messages: Sequence[ChatMessage]) -> str:
            acquired = lock.acquire(blocking=False)
            lock_free_during_generate["free"] = acquired
            if acquired:
                lock.release()
            return llm_json()

    result = diagnose(
        graph.connection,
        "叶片出现褐色病斑",
        llm_client=LockProbingLLM(),
        connection_lock=lock,
    )

    assert result.status is DiagnosisStatus.DIAGNOSED
    assert lock_free_during_generate["free"] is True
    # The pipeline must leave the caller-owned lock released after returning.
    assert lock.acquire(blocking=False) is True
    lock.release()


def test_connection_lock_guards_retrieval(monkeypatch):
    """MEDIUM-2b: retrieval itself must run while the lock is held."""

    import app.pipeline.diagnose  # noqa: F401  (ensure submodule is imported)

    diagnose_module = sys.modules["app.pipeline.diagnose"]

    lock = threading.Lock()
    held_during_retrieval = {}

    def fake_retrieve(connection, symptom_text, **kwargs):
        held_during_retrieval["locked"] = lock.acquire(blocking=False) is False
        return retrieve(connection, symptom_text, **kwargs)

    store = load_seed_graph()
    try:
        monkeypatch.setattr(diagnose_module, "retrieve", fake_retrieve)
        diagnose(
            store.connection,
            "叶片出现褐色病斑",
            llm_client=StubLLM(response=llm_json()),
            connection_lock=lock,
        )
    finally:
        store.close()

    assert held_during_retrieval["locked"] is True
