from __future__ import annotations

import json
import subprocess
import sys
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


def llm_json(
    *,
    diagnosis_ids: list[str] | None = None,
    suggestions: list[dict[str, object]] | None = None,
) -> str:
    return json.dumps(
        {
            "diagnosis": {
                "text": "检索证据支持稻瘟病诊断。",
                "referenced_entity_ids": diagnosis_ids
                or ["disease:rice-blast", "symptom:rice-blast-brown-lesions"],
            },
            "model_suggestions": suggestions
            if suggestions is not None
            else [
                {
                    "text": "可参考图谱中的抗病品种和肥水管理措施。",
                    "referenced_entity_ids": ["control:rice-blast-resistant-variety"],
                }
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


def test_grounding_filters_only_out_of_subgraph_suggestions(graph):
    safe_text = "可参考图谱中的抗病品种和肥水管理措施。"
    unsafe_text = "使用图谱外药剂甲。"
    llm = StubLLM(
        response=llm_json(
            suggestions=[
                {
                    "text": safe_text,
                    "referenced_entity_ids": ["control:rice-blast-resistant-variety"],
                },
                {
                    "text": unsafe_text,
                    "referenced_entity_ids": ["pesticide:not-in-subgraph"],
                },
            ]
        )
    )

    result = diagnose(graph.connection, "叶片出现褐色病斑", llm_client=llm)

    assert [item.text for item in result.model_suggestions] == [safe_text]
    assert result.grounding_rejections == 1
    assert unsafe_text not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_grounding_drops_an_ungrounded_diagnosis_but_keeps_verified_knowledge(graph):
    unsafe_text = "图谱外病害乙是最终诊断。"
    raw = json.loads(llm_json(diagnosis_ids=["disease:not-in-subgraph"]))
    raw["diagnosis"]["text"] = unsafe_text
    llm = StubLLM(response=json.dumps(raw, ensure_ascii=False))

    result = diagnose(graph.connection, "叶片出现褐色病斑", llm_client=llm)

    assert result.status is DiagnosisStatus.DIAGNOSED
    assert result.diagnosis is None
    assert result.verified_knowledge
    assert result.grounding_rejections == 1
    assert unsafe_text not in json.dumps(result.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "{}",
        json.dumps(
            {
                "diagnosis": {"text": "缺少引用", "referenced_entity_ids": []},
                "model_suggestions": [],
            },
            ensure_ascii=False,
        ),
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
