"""Create and load the small, provenance-carrying rice knowledge graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kuzu

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SEED_PATH = ROOT_DIR / "data" / "rice_seed_kg.json"


@dataclass
class GraphStore:
    """A Kuzu database and its connection kept alive as one retrieval unit."""

    database: kuzu.Database
    connection: kuzu.Connection

    def close(self) -> None:
        """Close the connection; Kuzu releases the database on object disposal."""

        self.connection.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


RELATION_FIELDS = ("source", "version", "confidence")


def load_seed_graph(
    database_path: str | Path = ":memory:",
    seed_path: str | Path = DEFAULT_SEED_PATH,
) -> GraphStore:
    """Create a fresh Kuzu graph and load the checked-in rice seed facts.

    The database path is intentionally a fresh path for this loader. Reusing a
    non-empty Kuzu directory raises Kuzu's schema error instead of silently
    appending duplicate facts.
    """

    seed_data = _read_seed_data(Path(seed_path))
    database = kuzu.Database(str(database_path))
    connection = kuzu.Connection(database)
    try:
        _create_schema(connection)
        _insert_seed_data(connection, seed_data)
    except Exception:
        connection.close()
        raise
    return GraphStore(database=database, connection=connection)


def _read_seed_data(seed_path: Path) -> dict[str, Any]:
    with seed_path.open("r", encoding="utf-8") as seed_file:
        payload = json.load(seed_file)
    _validate_seed_data(payload)
    return payload


def _validate_seed_data(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("crop") != "水稻":
        raise ValueError("seed data must describe 水稻")
    diseases = payload.get("diseases")
    if not isinstance(diseases, list) or not 3 <= len(diseases) <= 5:
        raise ValueError("seed data must contain 3 to 5 diseases or pests")

    node_ids: set[str] = set()
    for disease in diseases:
        _validate_node(disease, "disease", node_ids)
        symptoms = disease.get("symptoms")
        controls = disease.get("controls")
        if not symptoms or not controls:
            raise ValueError(f"{disease['id']} must have symptoms and controls")
        for symptom in symptoms:
            _validate_node(symptom, "symptom", node_ids)
            _validate_relation(symptom, f"{disease['id']} symptom relation")
        for control in controls:
            _validate_node(control, "control", node_ids)
            _validate_relation(control, f"{disease['id']} control relation")
            pesticides = control.get("pesticides")
            if not pesticides:
                raise ValueError(f"{control['id']} must have pesticides")
            for pesticide in pesticides:
                _validate_node(pesticide, "pesticide", node_ids)
                _validate_relation(pesticide, f"{control['id']} pesticide relation")


def _validate_node(node: Any, node_type: str, node_ids: set[str]) -> None:
    if not isinstance(node, dict) or not node.get("id") or not node.get("name"):
        raise ValueError(f"{node_type} must have id and name")
    node_id = node["id"]
    if node_id in node_ids:
        raise ValueError(f"duplicate seed node id: {node_id}")
    node_ids.add(node_id)


def _validate_relation(relation: Any, label: str) -> None:
    if not isinstance(relation, dict) or any(field not in relation for field in RELATION_FIELDS):
        raise ValueError(f"{label} must include source, version, and confidence")
    if not relation["source"] or not relation["version"]:
        raise ValueError(f"{label} provenance cannot be empty")
    if not isinstance(relation["confidence"], (float, int)) or not 0 <= relation["confidence"] <= 1:
        raise ValueError(f"{label} confidence must be between 0 and 1")


def _create_schema(connection: kuzu.Connection) -> None:
    statements = (
        """
        CREATE NODE TABLE Disease(
            id STRING PRIMARY KEY,
            name STRING,
            category STRING,
            crop STRING
        )
        """,
        """
        CREATE NODE TABLE Symptom(
            id STRING PRIMARY KEY,
            name STRING,
            crop STRING
        )
        """,
        """
        CREATE NODE TABLE ControlMethod(
            id STRING PRIMARY KEY,
            name STRING,
            description STRING,
            crop STRING
        )
        """,
        """
        CREATE NODE TABLE Pesticide(
            id STRING PRIMARY KEY,
            name STRING,
            active_ingredient STRING,
            safety_note STRING,
            crop STRING
        )
        """,
        """
        CREATE REL TABLE HAS_SYMPTOM(
            FROM Disease TO Symptom,
            source STRING,
            version STRING,
            confidence DOUBLE
        )
        """,
        """
        CREATE REL TABLE HAS_CONTROL(
            FROM Disease TO ControlMethod,
            source STRING,
            version STRING,
            confidence DOUBLE
        )
        """,
        """
        CREATE REL TABLE RECOMMENDS_PESTICIDE(
            FROM ControlMethod TO Pesticide,
            source STRING,
            version STRING,
            confidence DOUBLE
        )
        """,
    )
    for statement in statements:
        connection.execute(statement)


def _insert_seed_data(connection: kuzu.Connection, payload: dict[str, Any]) -> None:
    crop = payload["crop"]
    inserted_nodes: set[str] = set()
    for disease in payload["diseases"]:
        connection.execute(
            "CREATE (:Disease {id: $id, name: $name, category: $category, crop: $crop})",
            {
                "id": disease["id"],
                "name": disease["name"],
                "category": disease["category"],
                "crop": crop,
            },
        )
        inserted_nodes.add(disease["id"])
        for symptom in disease["symptoms"]:
            _create_node_once(connection, "Symptom", symptom, {"crop": crop}, inserted_nodes)
            _create_relation(connection, "HAS_SYMPTOM", disease["id"], symptom["id"], symptom)
        for control in disease["controls"]:
            _create_node_once(
                connection,
                "ControlMethod",
                control,
                {"description": control["description"], "crop": crop},
                inserted_nodes,
            )
            _create_relation(connection, "HAS_CONTROL", disease["id"], control["id"], control)
            for pesticide in control["pesticides"]:
                _create_node_once(
                    connection,
                    "Pesticide",
                    pesticide,
                    {
                        "active_ingredient": pesticide["active_ingredient"],
                        "safety_note": pesticide["safety_note"],
                        "crop": crop,
                    },
                    inserted_nodes,
                )
                _create_relation(
                    connection,
                    "RECOMMENDS_PESTICIDE",
                    control["id"],
                    pesticide["id"],
                    pesticide,
                )


def _create_node_once(
    connection: kuzu.Connection,
    table: str,
    node: dict[str, Any],
    extra_properties: dict[str, Any],
    inserted_nodes: set[str],
) -> None:
    if node["id"] in inserted_nodes:
        return
    properties = {"id": node["id"], "name": node["name"], **extra_properties}
    property_names = ", ".join(f"{key}: ${key}" for key in properties)
    connection.execute(f"CREATE (:{table} {{{property_names}}})", properties)
    inserted_nodes.add(node["id"])


def _create_relation(
    connection: kuzu.Connection,
    relation_type: str,
    from_id: str,
    to_id: str,
    provenance: dict[str, Any],
) -> None:
    connection.execute(
        f"""
        MATCH (from_node), (to_node)
        WHERE from_node.id = $from_id AND to_node.id = $to_id
        CREATE (from_node)-[:{relation_type} {{
            source: $source,
            version: $version,
            confidence: $confidence
        }}]->(to_node)
        """,
        {
            "from_id": from_id,
            "to_id": to_id,
            "source": provenance["source"],
            "version": provenance["version"],
            "confidence": float(provenance["confidence"]),
        },
    )
