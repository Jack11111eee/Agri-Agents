"""Deterministic orchestration for retrieval-grounded diagnosis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence, TypeVar

from app.kg.retrieval import RetrievalResult, RetrievalStatus, retrieve
from app.llm.protocol import LLMClient


class DiagnosisStatus(str, Enum):
    """Stable pipeline outcomes exposed to callers."""

    INVALID_INPUT = "INVALID_INPUT"
    ABSTAINED = "ABSTAINED"
    DIAGNOSED = "DIAGNOSED"


class LLMClientRequiredError(RuntimeError):
    """Raised when a retrieval hit has no generation client."""


class LLMOutputError(ValueError):
    """Raised when provider output violates the constrained JSON contract."""


@dataclass(frozen=True)
class DiagnosisStatement:
    text: str
    referenced_entity_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "referenced_entity_ids": list(self.referenced_entity_ids),
        }


@dataclass(frozen=True)
class ModelSuggestion:
    text: str
    referenced_entity_ids: tuple[str, ...]
    authoritative: bool = False
    label: str = "模型补充建议"

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "referenced_entity_ids": list(self.referenced_entity_ids),
            "authoritative": self.authoritative,
            "label": self.label,
        }


@dataclass(frozen=True)
class DiagnosisResult:
    status: DiagnosisStatus
    reason: str
    matched_symptoms: list[str] = field(default_factory=list)
    diagnosis: DiagnosisStatement | None = None
    verified_knowledge: list[dict[str, object]] = field(default_factory=list)
    model_suggestions: list[ModelSuggestion] = field(default_factory=list)
    evidence_chain: list[dict[str, object]] = field(default_factory=list)
    grounding_rejections: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "matched_symptoms": list(self.matched_symptoms),
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "verified_knowledge": [dict(item) for item in self.verified_knowledge],
            "model_suggestions": [item.to_dict() for item in self.model_suggestions],
            "evidence_chain": [dict(item) for item in self.evidence_chain],
            "grounding_rejections": self.grounding_rejections,
        }


def diagnose(
    connection: Any,
    symptom_text: object,
    *,
    llm_client: LLMClient | None = None,
) -> DiagnosisResult:
    """Retrieve evidence, short-circuit abstention, and ground LLM output.

    Invalid and abstained requests return before the LLM client is inspected.
    Provider and retrieval failures deliberately bubble so callers never receive
    model content disguised as a successful partial diagnosis.
    """

    if not isinstance(symptom_text, str) or not symptom_text.strip():
        return DiagnosisResult(
            status=DiagnosisStatus.INVALID_INPUT,
            reason="输入必须是非空文本",
        )

    retrieval = retrieve(connection, symptom_text)
    if retrieval.status is RetrievalStatus.ABSTAIN:
        return DiagnosisResult(
            status=DiagnosisStatus.ABSTAINED,
            reason=retrieval.reason,
            matched_symptoms=list(retrieval.matched_symptoms),
        )

    if llm_client is None:
        raise LLMClientRequiredError("检索命中后必须提供 LLM client")

    generated = llm_client.generate(_build_messages(retrieval))
    payload = _parse_generation(generated)
    diagnosis = _parse_diagnosis(payload["diagnosis"])
    suggestions = _parse_suggestions(payload["model_suggestions"])

    allowed_ids = frozenset(node.node_id for node in retrieval.subgraph.nodes)
    grounded_diagnosis, diagnosis_rejections = _ground_diagnosis(
        diagnosis,
        allowed_ids,
    )
    grounded_suggestions, suggestion_rejections = _ground_suggestions(
        suggestions,
        allowed_ids,
    )

    return DiagnosisResult(
        status=DiagnosisStatus.DIAGNOSED,
        reason=retrieval.reason,
        matched_symptoms=list(retrieval.matched_symptoms),
        diagnosis=grounded_diagnosis,
        verified_knowledge=[node.to_dict() for node in retrieval.subgraph.nodes],
        model_suggestions=grounded_suggestions,
        evidence_chain=[
            relationship.to_dict()
            for relationship in retrieval.subgraph.relationships
        ],
        grounding_rejections=diagnosis_rejections + suggestion_rejections,
    )


def _build_messages(retrieval: RetrievalResult) -> list[dict[str, str]]:
    subgraph = {
        "nodes": [node.to_dict() for node in retrieval.subgraph.nodes],
        "relationships": [
            relationship.to_dict()
            for relationship in retrieval.subgraph.relationships
        ],
        "allowed_entity_ids": sorted(
            node.node_id for node in retrieval.subgraph.nodes
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "你是农业诊断生成器，只能依据用户消息中的检索子图。"
                "只返回 JSON 对象，其中 diagnosis 是包含 text 与 "
                "referenced_entity_ids 的对象，model_suggestions 是同结构对象数组。"
                "所有引用必须来自 allowed_entity_ids，不得添加剂量、施药时机或子图外事实。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"subgraph": subgraph},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def _parse_generation(generated: object) -> Mapping[str, object]:
    if not isinstance(generated, str):
        raise LLMOutputError("LLM 输出必须是 JSON 字符串")

    try:
        payload = json.loads(generated)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMOutputError("LLM 返回非法 JSON") from exc

    if not isinstance(payload, dict):
        raise LLMOutputError("LLM JSON 顶层必须是对象")

    missing = {"diagnosis", "model_suggestions"} - payload.keys()
    if missing:
        raise LLMOutputError(f"LLM JSON 缺少字段: {', '.join(sorted(missing))}")

    return payload


def _parse_diagnosis(raw_item: object) -> DiagnosisStatement:
    return _parse_item(raw_item, "diagnosis", DiagnosisStatement)


def _parse_suggestions(raw_items: object) -> list[ModelSuggestion]:
    if not isinstance(raw_items, list):
        raise LLMOutputError("LLM JSON 字段 model_suggestions 必须是数组")
    return [
        _parse_item(item, f"model_suggestions[{index}]", ModelSuggestion)
        for index, item in enumerate(raw_items)
    ]


GeneratedItem = TypeVar("GeneratedItem", DiagnosisStatement, ModelSuggestion)


def _parse_item(
    raw_item: object,
    field: str,
    item_type: type[GeneratedItem],
) -> GeneratedItem:
    if not isinstance(raw_item, dict):
        raise LLMOutputError(f"LLM JSON 字段 {field} 必须是对象")

    text = raw_item.get("text")
    entity_ids = raw_item.get("referenced_entity_ids")
    if not isinstance(text, str) or not text.strip():
        raise LLMOutputError(f"{field}.text 必须是非空字符串")
    if (
        not isinstance(entity_ids, list)
        or not entity_ids
        or any(
            not isinstance(entity_id, str) or not entity_id
            for entity_id in entity_ids
        )
    ):
        raise LLMOutputError(
            f"{field}.referenced_entity_ids 必须是非空字符串数组"
        )

    return item_type(
        text=text.strip(),
        referenced_entity_ids=tuple(entity_ids),
    )


def _ground_diagnosis(
    diagnosis: DiagnosisStatement,
    allowed_ids: frozenset[str],
) -> tuple[DiagnosisStatement | None, int]:
    if _is_grounded(diagnosis, allowed_ids):
        return diagnosis, 0
    return None, 1


def _ground_suggestions(
    suggestions: Sequence[ModelSuggestion],
    allowed_ids: frozenset[str],
) -> tuple[list[ModelSuggestion], int]:
    grounded = [
        suggestion
        for suggestion in suggestions
        if _is_grounded(suggestion, allowed_ids)
    ]
    return grounded, len(suggestions) - len(grounded)


def _is_grounded(
    item: DiagnosisStatement | ModelSuggestion,
    allowed_ids: frozenset[str],
) -> bool:
    return set(item.referenced_entity_ids).issubset(allowed_ids)
