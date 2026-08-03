"""Fail if a completed GSD task's claimed key files are not in HEAD.

A task marked ``complete`` in ``.gsd/gsd.db`` only records what an agent
*claims* it delivered. If the agent never committed, the claim and the tree
disagree, and GSD's security review — which reads committed code, then
restores the tree to prove it changed nothing — silently discards the work.

Run this at the task-completion boundary: after the commit, before resuming
auto-mode or dispatching a review. Not part of the verification command
chain, because the tree is legitimately dirty while a task is in progress.

Usage:
    python scripts/check_task_committed.py                 # newest complete task
    python scripts/check_task_committed.py --task M001/S02/T05
    python scripts/check_task_committed.py --all            # every complete task
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GSD_DB = PROJECT_ROOT / ".gsd" / "gsd.db"

# .gsd/ is gitignored, so GSD's own artifacts can never be "committed".
# Only check paths git is allowed to track.
IGNORED_PREFIXES = (".gsd/",)


class GuardError(RuntimeError):
    """Raised when the guard cannot inspect the task or the repository."""


def load_tasks(task_ref: str | None, check_all: bool) -> list[dict[str, object]]:
    """Read completed tasks and their claimed key files from the GSD database."""

    if not GSD_DB.exists():
        raise GuardError(f"未找到 GSD 数据库: {GSD_DB}")

    query = (
        "select milestone_id, slice_id, id, key_files, completed_at "
        "from tasks where status = 'complete'"
    )
    params: tuple[str, ...] = ()
    if task_ref:
        parts = task_ref.split("/")
        if len(parts) != 3:
            raise GuardError(f"--task 需要 M###/S##/T## 形式，收到: {task_ref}")
        query += " and milestone_id = ? and slice_id = ? and id = ?"
        params = tuple(parts)

    # completed_at is an ISO-8601 string, so lexical ordering is chronological.
    query += " order by completed_at desc"

    connection = sqlite3.connect(f"file:{GSD_DB}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(query, params)]
    finally:
        connection.close()

    if not rows:
        target = task_ref or "any task"
        raise GuardError(f"数据库中没有状态为 complete 的任务: {target}")

    return rows if (check_all or task_ref) else rows[:1]


def parse_key_files(raw: object) -> list[str]:
    """Return the checkable key files, dropping gitignored GSD artifacts."""

    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise GuardError(f"key_files 不是合法 JSON: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise GuardError(f"key_files 必须是数组，收到: {parsed!r}")
    return [
        path
        for path in parsed
        if isinstance(path, str)
        and path
        and not path.startswith(IGNORED_PREFIXES)
    ]


def git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GuardError(f"git {' '.join(args)} 失败: {completed.stderr.strip()}")
    return completed.stdout


def uncommitted_paths(paths: list[str]) -> dict[str, str]:
    """Map each key file with pending changes to its porcelain status code."""

    if not paths:
        return {}
    # --porcelain keeps the two-column status stable across git versions.
    output = git("status", "--porcelain", "--untracked-files=all", "--", *paths)
    pending: dict[str, str] = {}
    for line in output.splitlines():
        if len(line) > 3:
            pending[line[3:].strip().strip('"')] = line[:2]
    return pending


def missing_from_head(paths: list[str]) -> list[str]:
    """Return key files absent from HEAD, so a claim can't point at nothing."""

    if not paths:
        return []
    tracked = set(git("ls-tree", "-r", "--name-only", "HEAD", "--", *paths).split("\n"))
    return [path for path in paths if path not in tracked]


def check_task(task: dict[str, object]) -> list[str]:
    """Return human-readable problems for one completed task."""

    ref = f"{task['milestone_id']}/{task['slice_id']}/{task['id']}"
    key_files = parse_key_files(task.get("key_files"))
    if not key_files:
        print(f"  {ref}: 无可校验的 key_files，跳过")
        return []

    problems: list[str] = []
    for path, code in sorted(uncommitted_paths(key_files).items()):
        problems.append(f"{ref}: {path} 有未提交改动（git status: {code}）")
    for path in missing_from_head(key_files):
        problems.append(f"{ref}: {path} 不在 HEAD 中")

    if not problems:
        print(f"  {ref}: {len(key_files)} 个 key_files 均已在 HEAD ✓")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="校验已完成 GSD 任务声称的 key_files 确实已提交进 HEAD。",
    )
    parser.add_argument("--task", help="限定单个任务，形如 M001/S02/T05")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="check_all",
        help="校验所有状态为 complete 的任务",
    )
    args = parser.parse_args()

    try:
        tasks = load_tasks(args.task, args.check_all)
        print(f"校验 {len(tasks)} 个已完成任务的 key_files 提交状态：")
        problems = [problem for task in tasks for problem in check_task(task)]
    except GuardError as error:
        print(f"守卫无法执行: {error}", file=sys.stderr)
        return 2

    if problems:
        # Keep the per-task lines above the summary when streams are redirected.
        sys.stdout.flush()
        print("\n任务声称的交付物未进入 HEAD：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\n先提交这些改动再恢复 auto-mode 或派发审查。"
            "安全审查只读已提交代码，并会在结束时恢复工作树，"
            "未提交的改动会被静默丢弃。",
            file=sys.stderr,
        )
        return 1

    print("\n全部通过：任务声称的交付物都在 HEAD 中。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
