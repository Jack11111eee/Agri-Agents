from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.api.main import _LazyLLMClient, create_app
from app.llm.protocol import ChatMessage

EXPECTED_RESPONSE_FIELDS = {
    "status",
    "reason",
    "matched_symptoms",
    "diagnosis",
    "verified_knowledge",
    "model_suggestions",
    "evidence_chain",
    "grounding_rejections",
}


@dataclass
class StubLLM:
    response: str = ""
    error: Exception | None = None
    calls: list[list[ChatMessage]] = field(default_factory=list)
    close_calls: int = 0

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.close_calls += 1


def grounded_response() -> str:
    """A provider response under the ID-only contract (D006)."""

    return json.dumps(
        {
            "model_suggestions": [
                {"referenced_entity_ids": ["control:rice-blast-resistant-variety"]}
            ],
        },
        ensure_ascii=False,
    )


def test_post_diagnose_returns_stable_diagnosed_contract() -> None:
    llm = StubLLM(response=grounded_response())
    application = create_app(llm_client_factory=lambda: llm)

    with TestClient(application) as client:
        response = client.post("/diagnose", json={"symptoms": "叶片出现褐色病斑"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == EXPECTED_RESPONSE_FIELDS
    assert payload["status"] == "DIAGNOSED"
    assert payload["diagnosis"]["text"] == "图谱匹配到可能相关病害：稻瘟病"
    assert payload["verified_knowledge"]
    assert payload["model_suggestions"] == [
        {
            "text": (
                "图谱防治措施：选用抗病品种并加强肥水管理；"
                "结合品种抗性和田间管理降低病害发生风险。"
            ),
            "referenced_entity_ids": ["control:rice-blast-resistant-variety"],
            "authoritative": False,
            "label": "模型补充建议",
        }
    ]
    assert llm.calls
    for relationship in payload["evidence_chain"]:
        assert {"source", "version", "confidence"} <= relationship["properties"].keys()


def test_lifespan_reuses_and_closes_initialized_llm_client() -> None:
    llm = StubLLM(response=grounded_response())
    factory_calls = 0

    def llm_factory() -> StubLLM:
        nonlocal factory_calls
        factory_calls += 1
        return llm

    application = create_app(llm_client_factory=llm_factory)
    with TestClient(application) as client:
        first = client.post("/diagnose", json={"symptoms": "叶片出现褐色病斑"})
        second = client.post("/diagnose", json={"symptoms": "叶片出现褐色病斑"})

        assert first.status_code == 200
        assert second.status_code == 200
        assert factory_calls == 1
        assert len(llm.calls) == 2
        assert llm.close_calls == 0

    assert llm.close_calls == 1


def test_post_diagnose_abstains_without_constructing_llm() -> None:
    factory_calls = 0

    def forbidden_factory() -> StubLLM:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("ABSTAINED request constructed an LLM client")

    application = create_app(llm_client_factory=forbidden_factory)
    with TestClient(application) as client:
        response = client.post("/diagnose", json={"symptoms": "不存在的症状XYZ"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == EXPECTED_RESPONSE_FIELDS
    assert payload["status"] == "ABSTAINED"
    assert payload["reason"]
    assert payload["diagnosis"] is None
    assert payload["verified_knowledge"] == []
    assert payload["model_suggestions"] == []
    assert payload["evidence_chain"] == []
    assert factory_calls == 0


def test_post_diagnose_rejects_blank_input_without_constructing_llm() -> None:
    factory_calls = 0

    def forbidden_factory() -> StubLLM:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("INVALID_INPUT request constructed an LLM client")

    application = create_app(llm_client_factory=forbidden_factory)
    with TestClient(application) as client:
        response = client.post("/diagnose", json={"symptoms": "   "})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == EXPECTED_RESPONSE_FIELDS
    assert payload["status"] == "INVALID_INPUT"
    assert payload["reason"] == "输入必须是非空文本"
    assert payload["matched_symptoms"] == []
    assert factory_calls == 0


def test_post_diagnose_uses_422_for_malformed_request_body() -> None:
    application = create_app(llm_client_factory=lambda: StubLLM())

    with TestClient(application) as client:
        missing = client.post("/diagnose", json={})
        wrong_type = client.post("/diagnose", json={"symptoms": 42})

    assert missing.status_code == 422
    assert wrong_type.status_code == 422


def test_post_diagnose_rejects_oversized_symptoms_without_constructing_llm() -> None:
    factory_calls = 0

    def forbidden_factory() -> StubLLM:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("oversized request constructed an LLM client")

    application = create_app(llm_client_factory=forbidden_factory)
    with TestClient(application) as client:
        response = client.post("/diagnose", json={"symptoms": "症" * 2001})

    # MEDIUM-2a: an unbounded symptom string must be rejected server-side
    # before it reaches retrieval or the provider.
    assert response.status_code == 422
    assert factory_calls == 0


def test_lazy_llm_client_constructs_once_under_concurrent_first_hits() -> None:
    """MEDIUM-2b follow-up: narrowing the retrieval lock exposed lazy init.

    With generate() no longer serialized by the retrieval lock, concurrent
    first hits must not each build a client and orphan a transport. The
    construction lock must keep factory calls at exactly one.
    """

    factory_calls = 0
    factory_guard = threading.Lock()
    start = threading.Barrier(8)

    def slow_factory() -> StubLLM:
        nonlocal factory_calls
        with factory_guard:
            factory_calls += 1
        # Widen the race window so a missing lock would reliably double-build.
        import time

        time.sleep(0.02)
        return StubLLM(response=grounded_response())

    lazy = _LazyLLMClient(slow_factory)
    errors: list[Exception] = []

    def hit() -> None:
        try:
            start.wait(timeout=5)
            lazy.generate([{"role": "user", "content": "x"}])
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert factory_calls == 1
    assert lazy.initialized is True


def test_post_diagnose_reports_provider_configuration_failure() -> None:
    def failing_factory() -> StubLLM:
        raise RuntimeError("provider configuration failed")

    application = create_app(llm_client_factory=failing_factory)
    with TestClient(application) as client:
        response = client.post("/diagnose", json={"symptoms": "叶片出现褐色病斑"})

    assert response.status_code == 503
    assert response.json() == {"detail": "诊断生成服务暂不可用"}


def test_post_diagnose_reports_provider_and_format_failures() -> None:
    provider_failure = create_app(
        llm_client_factory=lambda: StubLLM(error=TimeoutError("provider timed out"))
    )
    malformed_response = create_app(llm_client_factory=lambda: StubLLM(response="not-json"))

    with TestClient(provider_failure) as client:
        provider_response = client.post(
            "/diagnose",
            json={"symptoms": "叶片出现褐色病斑"},
        )
    with TestClient(malformed_response) as client:
        malformed = client.post(
            "/diagnose",
            json={"symptoms": "叶片出现褐色病斑"},
        )

    assert provider_response.status_code == 502
    assert provider_response.json() == {"detail": "诊断生成服务返回无效响应"}
    assert malformed.status_code == 502
    assert malformed.json() == {"detail": "诊断生成服务返回无效响应"}


def test_graph_startup_failure_bubbles() -> None:
    def failing_graph_store_factory():
        raise OSError("graph seed unavailable")

    application = create_app(graph_store_factory=failing_graph_store_factory)

    with pytest.raises(OSError, match="graph seed unavailable"):
        with TestClient(application):
            pass


def test_static_page_is_served_without_shadowing_diagnose_route() -> None:
    llm = StubLLM(response=grounded_response())
    application = create_app(llm_client_factory=lambda: llm)

    with TestClient(application) as client:
        page = client.get("/")
        api = client.post("/diagnose", json={"symptoms": "不存在的症状XYZ"})

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert 'id="diagnosis-form"' in page.text
    assert 'data-status="IDLE"' in page.text
    assert api.status_code == 200
    assert api.json()["status"] == "ABSTAINED"
