"""Deterministic orchestration for retrieval-grounded diagnosis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from app.kg.retrieval import GraphNode, RetrievalResult, RetrievalStatus, retrieve
from app.llm.protocol import LLMClient

# The provider may only pick control/pesticide entities from the retrieved
# subgraph. Every user-visible string is rendered by this module from stored
# graph properties, so provider text can never carry a safety-sensitive claim.
SELECTABLE_NODE_TYPES = ("ControlMethod", "Pesticide")


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
class GeneratedSelection:
    """The only thing a provider is allowed to return: graph entity IDs."""

    referenced_entity_ids: tuple[str, ...]


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
    """Retrieve evidence, short-circuit abstention, and render graph-only text.

    Invalid and abstained requests return before the LLM client is inspected.
    On a hit the provider may only select graph entity IDs; the diagnosis and
    every suggestion string are rendered here from stored subgraph properties.
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
    selections = _parse_selections(_parse_generation(generated)["model_suggestions"])

    selectable_nodes = {
        node.node_id: node
        for node in retrieval.subgraph.nodes
        if node.node_type in SELECTABLE_NODE_TYPES
    }
    suggestions, rejections = _render_suggestions(selections, selectable_nodes)

    return DiagnosisResult(
        status=DiagnosisStatus.DIAGNOSED,
        reason=retrieval.reason,
        matched_symptoms=list(retrieval.matched_symptoms),
        diagnosis=_build_diagnosis(retrieval),
        verified_knowledge=[node.to_dict() for node in retrieval.subgraph.nodes],
        model_suggestions=suggestions,
        evidence_chain=[
            relationship.to_dict()
            for relationship in retrieval.subgraph.relationships
        ],
        grounding_rejections=rejections,
    )


def _build_messages(retrieval: RetrievalResult) -> list[dict[str, str]]:
    subgraph = {
        "nodes": [node.to_dict() for node in retrieval.subgraph.nodes],
        "relationships": [
            relationship.to_dict()
            for relationship in retrieval.subgraph.relationships
        ],
        "allowed_entity_ids": sorted(
            node.node_id
            for node in retrieval.subgraph.nodes
            if node.node_type in SELECTABLE_NODE_TYPES
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "你是农业诊断的防治措施选择器，只能依据用户消息中的检索子图。"
                "只返回 JSON 对象，唯一允许的顶层字段是 model_suggestions；"
                "数组每项唯一允许的字段是 "
                "referenced_entity_ids，且至少引用一个 allowed_entity_ids 中的 "
                "ControlMethod 或 Pesticide 实体。不得返回 diagnosis、text、剂量、"
                "施药时机、安全间隔期或任何子图外事实；展示文本由服务器生成。"
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

    # A diagnosis or free-text field is a contract violation, not something to
    # filter: accepting it would let valid IDs launder an unsupported claim.
    unexpected = payload.keys() - {"model_suggestions"}
    if unexpected:
        raise LLMOutputError(
            f"LLM JSON 含未授权顶层字段: {', '.join(sorted(unexpected))}"
        )
    if "model_suggestions" not in payload:
        raise LLMOutputError("LLM JSON 缺少字段: model_suggestions")

    return payload


def _parse_selections(raw_items: object) -> list[GeneratedSelection]:
    if not isinstance(raw_items, list):
        raise LLMOutputError("LLM JSON 字段 model_suggestions 必须是数组")
    return [
        _parse_selection(item, f"model_suggestions[{index}]")
        for index, item in enumerate(raw_items)
    ]


def _parse_selection(raw_item: object, field: str) -> GeneratedSelection:
    if not isinstance(raw_item, dict):
        raise LLMOutputError(f"LLM JSON 字段 {field} 必须是对象")

    allowed_fields = {"referenced_entity_ids"}
    unexpected = raw_item.keys() - allowed_fields
    if unexpected:
        raise LLMOutputError(
            f"{field} 含未授权字段: {', '.join(sorted(unexpected))}"
        )

    entity_ids = raw_item.get("referenced_entity_ids")
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

    return GeneratedSelection(referenced_entity_ids=tuple(entity_ids))


def _build_diagnosis(retrieval: RetrievalResult) -> DiagnosisStatement:
    disease_nodes = [
        node for node in retrieval.subgraph.nodes if node.node_type == "Disease"
    ]
    names = "、".join(str(node.properties.get("name", "")) for node in disease_nodes)
    return DiagnosisStatement(
        text=f"图谱匹配到可能相关病害：{names}",
        referenced_entity_ids=tuple(node.node_id for node in disease_nodes),
    )


def _render_suggestions(
    selections: Sequence[GeneratedSelection],
    selectable_nodes: Mapping[str, GraphNode],
) -> tuple[list[ModelSuggestion], int]:
    suggestions: list[ModelSuggestion] = []
    rejections = 0
    for selection in selections:
        nodes = [
            selectable_nodes[entity_id]
            for entity_id in selection.referenced_entity_ids
            if entity_id in selectable_nodes
        ]
        # Any out-of-subgraph or non-selectable ID invalidates the whole item;
        # partially rendering it would imply support the graph does not give.
        if len(nodes) != len(selection.referenced_entity_ids) or not nodes:
            rejections += 1
            continue
        suggestions.append(
            ModelSuggestion(
                text="；".join(_render_node(node) for node in nodes),
                referenced_entity_ids=selection.referenced_entity_ids,
            )
        )
    return suggestions, rejections


def _render_node(node: GraphNode) -> str:
    """Render one graph node using only its stored properties."""

    properties = node.properties
    name = str(properties.get("name", "")).strip()
    if node.node_type == "Pesticide":
        parts = [f"图谱登记药剂：{name}"]
        ingredient = str(properties.get("active_ingredient", "")).strip()
        if ingredient:
            parts.append(f"有效成分：{ingredient}")
        safety_note = str(properties.get("safety_note", "")).strip()
        if safety_note:
            parts.append(f"安全提示：{safety_note}")
        return "；".join(parts)

    parts = [f"图谱防治措施：{name}"]
    description = str(properties.get("description", "")).strip()
    if description:
        parts.append(description)
    return "；".join(parts)
