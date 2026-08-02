"""Print deterministic rice knowledge-graph retrieval results as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``python scripts/query_kg.py`` work from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.loader import load_seed_graph  # noqa: E402
from app.kg.retrieval import retrieve  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query the seeded rice knowledge graph by symptom feature."
    )
    parser.add_argument(
        "--symptoms",
        required=True,
        help="症状特征文本；检索参数将由 app.kg.retrieval 绑定到 Cypher。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="显式请求 JSON 输出；脚本默认也输出 JSON。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with load_seed_graph() as graph:
        result = retrieve(graph.connection, args.symptoms)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
