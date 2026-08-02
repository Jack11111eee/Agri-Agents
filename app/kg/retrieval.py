"""Deterministic, LLM-free symptom retrieval over the Kuzu seed graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import kuzu

DEFAULT_MIN_MATCHED_SYMPTOMS = 1

# User input is always bound as $symptom_feature. Keeping the query constant
# makes it easy to inspect and prevents symptom text from becoming Cypher.
RETRIEVAL_CYPHER = """
MATCH (d:Disease)-[hs:HAS_SYMPTOM]->(s:Symptom),
      (d)-[hc:HAS_CONTROL]->(m:ControlMethod),
      (m)-[rp:RECOMMENDS_PESTICIDE]->(p:Pesticide)
WHERE CONTAINS($symptom_feature, s.name)
   OR CONTAINS(s.name, $symptom_feature)
RETURN
    d.id, d.name, d.category, d.crop,
    s.id, s.name, s.crop,
    m.id, m.name, m.description, m.crop,
    p.id, p.name, p.active_ingredient, p.safety_note, p.crop,
    hs.source, hs.version, hs.confidence,
    hc.source, hc.version, hc.confidence,
    rp.source, rp.version, rp.confidence
"""


class RetrievalStatus(str, Enum):
    HIT = "HIT"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    properties: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.node_id, "type": self.node_type, "properties": dict(self.properties)}


@dataclass(frozen=True)
class GraphRelationship:
    relation_type: str
    source_id: str
    target_id: str
    properties: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.relation_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": dict(self.properties),
        }


@dataclass(frozen=True)
class Subgraph:
    nodes: list[GraphNode] = field(default_factory=list)
    relationships: list[GraphRelationship] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
        }


@dataclass(frozen=True)
class RetrievalResult:
    status: RetrievalStatus
    reason: str
    matched_symptoms: list[str]
    subgraph: Subgraph = field(default_factory=Subgraph)

    @property
    def is_hit(self) -> bool:
        return self.status is RetrievalStatus.HIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "matched_symptoms": list(self.matched_symptoms),
            "subgraph": self.subgraph.to_dict(),
        }


def retrieve(
    connection: kuzu.Connection,
    symptoms: str | Sequence[str],
    *,
    min_matched_symptoms: int = DEFAULT_MIN_MATCHED_SYMPTOMS,
) -> RetrievalResult:
    """Retrieve a provenance-carrying subgraph or return an explicit abstention.

    A text query is split only when the caller supplies a sequence. Matching is
    performed by the parameterized Cypher query above; no model or network
    client is imported or called anywhere in this module.
    """

    if min_matched_symptoms < 1:
        raise ValueError("min_matched_symptoms must be at least 1")
    features = _normalize_symptoms(symptoms)
    if not features:
        return RetrievalResult(
            status=RetrievalStatus.ABSTAIN,
            reason="未提供可检索的症状特征",
            matched_symptoms=[],
        )

    candidates: dict[str, dict[str, Any]] = {}
    for feature in features:
        query_result = connection.execute(RETRIEVAL_CYPHER, {"symptom_feature": feature})
        for row in query_result.get_all():
            _accumulate_candidate(candidates, row)

    matched_symptoms = _ordered_matched_symptoms(candidates)
    eligible = {
        disease_id: candidate
        for disease_id, candidate in candidates.items()
        if len(candidate["matched_symptoms"]) >= min_matched_symptoms
    }
    if not eligible:
        if not candidates:
            reason = "图谱中未找到与输入症状相匹配的事实"
        else:
            max_matches = max(
                len(candidate["matched_symptoms"]) for candidate in candidates.values()
            )
            reason = (
                f"匹配症状数不足：至少需要 {min_matched_symptoms} 个，"
                f"当前候选最多匹配 {max_matches} 个"
            )
        return RetrievalResult(
            status=RetrievalStatus.ABSTAIN,
            reason=reason,
            matched_symptoms=matched_symptoms,
        )

    return RetrievalResult(
        status=RetrievalStatus.HIT,
        reason=(
            f"命中 {len(eligible)} 个候选实体，"
            f"匹配症状 {len(matched_symptoms)} 个（阈值 {min_matched_symptoms}）"
        ),
        matched_symptoms=matched_symptoms,
        subgraph=_build_subgraph(eligible),
    )


def _normalize_symptoms(symptoms: str | Sequence[str]) -> list[str]:
    if isinstance(symptoms, str):
        raw_features = [symptoms]
    elif isinstance(symptoms, Sequence) and not isinstance(symptoms, (bytes, bytearray)):
        raw_features = list(symptoms)
    else:
        raise TypeError("symptoms must be a string or a sequence of strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for feature in raw_features:
        if not isinstance(feature, str):
            raise TypeError("each symptom feature must be a string")
        feature = feature.strip()
        if feature and feature not in seen:
            normalized.append(feature)
            seen.add(feature)
    return normalized


def _accumulate_candidate(candidates: dict[str, dict[str, Any]], row: list[Any]) -> None:
    disease_id = row[0]
    candidate = candidates.setdefault(disease_id, {"matched_symptoms": [], "rows": []})
    symptom_name = row[5]
    if symptom_name not in candidate["matched_symptoms"]:
        candidate["matched_symptoms"].append(symptom_name)
    if row not in candidate["rows"]:
        candidate["rows"].append(row)


def _ordered_matched_symptoms(candidates: dict[str, dict[str, Any]]) -> list[str]:
    matched: list[str] = []
    for candidate in candidates.values():
        for symptom in candidate["matched_symptoms"]:
            if symptom not in matched:
                matched.append(symptom)
    return matched


def _build_subgraph(eligible: dict[str, dict[str, Any]]) -> Subgraph:
    nodes: dict[str, GraphNode] = {}
    relationships: dict[tuple[str, str, str], GraphRelationship] = {}
    for candidate in eligible.values():
        for row in candidate["rows"]:
            _add_node(
                nodes, row[0], "Disease", {"name": row[1], "category": row[2], "crop": row[3]}
            )
            _add_node(nodes, row[4], "Symptom", {"name": row[5], "crop": row[6]})
            _add_node(
                nodes,
                row[7],
                "ControlMethod",
                {"name": row[8], "description": row[9], "crop": row[10]},
            )
            _add_node(
                nodes,
                row[11],
                "Pesticide",
                {
                    "name": row[12],
                    "active_ingredient": row[13],
                    "safety_note": row[14],
                    "crop": row[15],
                },
            )
            _add_relationship(relationships, "HAS_SYMPTOM", row[0], row[4], row[16:19])
            _add_relationship(relationships, "HAS_CONTROL", row[0], row[7], row[19:22])
            _add_relationship(relationships, "RECOMMENDS_PESTICIDE", row[7], row[11], row[22:25])
    return Subgraph(nodes=list(nodes.values()), relationships=list(relationships.values()))


def _add_node(
    nodes: dict[str, GraphNode], node_id: str, node_type: str, properties: dict[str, Any]
) -> None:
    nodes.setdefault(
        node_id, GraphNode(node_id=node_id, node_type=node_type, properties=properties)
    )


def _add_relationship(
    relationships: dict[tuple[str, str, str], GraphRelationship],
    relation_type: str,
    source_id: str,
    target_id: str,
    provenance: Sequence[Any],
) -> None:
    key = (relation_type, source_id, target_id)
    relationships.setdefault(
        key,
        GraphRelationship(
            relation_type=relation_type,
            source_id=source_id,
            target_id=target_id,
            properties={
                "source": provenance[0],
                "version": provenance[1],
                "confidence": provenance[2],
            },
        ),
    )
