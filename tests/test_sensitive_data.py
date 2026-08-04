"""R010 专项验证：敏感数据不外泄。

覆盖三条外泄通道：
1. 出站 —— 原始症状文本（可能含 PII）不得进入发往供应商的 prompt。
2. 响应 —— API key、内部异常细节不得出现在 HTTP 响应体中。
3. 日志 —— 症状文本与 API key 不得写入服务日志。

测试只使用合成哨兵值，不把真实个人联系方式或凭据写入 fixture。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.kg.loader import load_seed_graph
from app.llm.protocol import ChatMessage
from app.pipeline.diagnose import DiagnosisStatus, diagnose

# 合成哨兵值用于追踪边界，避免把真实个人数据或 key 形状的凭据写入测试。
SENSITIVE_SYMPTOM = (
    "叶片出现褐色病斑；NAME_SENTINEL_R010；PHONE_SENTINEL_R010；"
    "ADDRESS_SENTINEL_R010；EMAIL_SENTINEL_R010"
)
SENSITIVE_FRAGMENTS = (
    "NAME_SENTINEL_R010",
    "PHONE_SENTINEL_R010",
    "ADDRESS_SENTINEL_R010",
    "EMAIL_SENTINEL_R010",
)
API_KEY_SENTINEL = "API_KEY_SENTINEL_R010"
INTERNAL_ERROR_SENTINEL = "INTERNAL_ERROR_SENTINEL_R010"


@dataclass
class RecordingLLM:
    response: str = '{"model_suggestions": []}'
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return self.response


@dataclass
class FailingLLM:
    error: Exception

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        raise self.error


@pytest.fixture
def graph():
    store = load_seed_graph()
    yield store
    store.close()


def test_sensitive_symptom_is_removed_from_outbound_llm_prompt(graph):
    llm = RecordingLLM()

    result = diagnose(graph.connection, SENSITIVE_SYMPTOM, llm_client=llm)

    assert result.status is DiagnosisStatus.DIAGNOSED
    assert len(llm.calls) == 1
    outbound_prompt = json.dumps(llm.calls[0], ensure_ascii=False)
    assert "叶片出现褐色病斑" in outbound_prompt
    assert all(fragment not in outbound_prompt for fragment in SENSITIVE_FRAGMENTS)


def test_provider_configuration_failure_does_not_echo_key_or_symptom(
    caplog,
):
    failure = RuntimeError(
        f"provider configuration failed: {API_KEY_SENTINEL}; "
        f"request={SENSITIVE_FRAGMENTS[0]}"
    )

    def failing_factory() -> FailingLLM:
        raise failure

    application = create_app(llm_client_factory=failing_factory)
    with caplog.at_level(logging.INFO, logger="app.api.main"):
        with TestClient(application) as client:
            response = client.post("/diagnose", json={"symptoms": SENSITIVE_SYMPTOM})

    assert response.status_code == 503
    assert response.json() == {"detail": "诊断生成服务暂不可用"}
    observable_text = response.text + caplog.text
    assert API_KEY_SENTINEL not in observable_text
    assert all(fragment not in observable_text for fragment in SENSITIVE_FRAGMENTS)


def test_provider_failure_does_not_echo_key_internal_details_or_symptom(
    caplog,
):
    failure = RuntimeError(
        f"provider failure: {API_KEY_SENTINEL}; "
        f"details={INTERNAL_ERROR_SENTINEL}; request={SENSITIVE_FRAGMENTS[0]}"
    )
    application = create_app(
        llm_client_factory=lambda: FailingLLM(error=failure),
    )

    with caplog.at_level(logging.INFO, logger="app.api.main"):
        with TestClient(application) as client:
            response = client.post("/diagnose", json={"symptoms": SENSITIVE_SYMPTOM})

    assert response.status_code == 502
    assert response.json() == {"detail": "诊断生成服务返回无效响应"}
    observable_text = response.text + caplog.text
    assert API_KEY_SENTINEL not in observable_text
    assert INTERNAL_ERROR_SENTINEL not in observable_text
    assert all(fragment not in observable_text for fragment in SENSITIVE_FRAGMENTS)
