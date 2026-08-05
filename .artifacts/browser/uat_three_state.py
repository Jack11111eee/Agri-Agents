"""Browser UAT for M001 S02 three-state diagnosis flow.

Matches 01-02-UAT.md: hit / abstain / invalid-input flows against a live
`uvicorn app.api.main:app` server. Emits one screenshot per flow plus a
structured JSON summary of assertions and captured /diagnose responses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = Path(__file__).resolve().parent

HIT_INPUT = "叶片出现褐色病斑"
ABSTAIN_INPUT = "不存在的症状XYZ"

TOP_LEVEL_FIELDS = {
    "status",
    "reason",
    "matched_symptoms",
    "diagnosis",
    "verified_knowledge",
    "model_suggestions",
    "evidence_chain",
    "grounding_rejections",
}

results: dict = {"checks": [], "responses": {}, "ok": True}


def record(name: str, passed: bool, detail: str = "") -> None:
    results["checks"].append({"name": name, "passed": passed, "detail": detail})
    results["ok"] = results["ok"] and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        diagnoses_responses: list[tuple[int, dict]] = []

        def on_response(resp):
            if resp.url.endswith("/diagnose") and resp.request.method == "POST":
                try:
                    diagnoses_responses.append((resp.status, resp.json()))
                except Exception:
                    diagnoses_responses.append((resp.status, None))

        page.on("response", on_response)

        # ── Smoke: page renders ──
        page.goto(BASE_URL, wait_until="networkidle")
        expect(page).to_have_title("稻作诊断台")
        expect(page.locator("#symptoms")).to_be_visible()
        expect(page.locator("#submit-button")).to_be_visible()
        expect(page.locator("#result")).to_have_attribute("data-status", "IDLE")
        record("smoke_page_renders", True, "title=稻作诊断台, status=IDLE")

        # ── Flow 1: HIT ──
        page.locator("#symptoms").fill(HIT_INPUT)
        page.locator("#submit-button").click()
        expect(page.locator("#result")).to_have_attribute(
            "data-status", "DIAGNOSED", timeout=60000
        )
        expect(page.locator(".diagnosis-heading h2")).not_to_be_empty()
        expect(page.locator(".knowledge-grid .knowledge-item").first).to_be_visible()
        expect(page.locator(".suggestion-item").first).to_be_visible()
        expect(page.locator(".suggestion-item .authority-note").first).to_contain_text(
            "模型生成，非权威处方"
        )
        evidence_rows = page.locator(".evidence-wrap table tbody tr")
        row_count = evidence_rows.count()
        record("hit_diagnosed", True, f"evidence_rows={row_count}")
        record("hit_evidence_rows", row_count > 0, f"count={row_count}")
        headers = page.locator(".evidence-wrap table thead th").all_inner_texts()
        record(
            "hit_evidence_columns",
            all(h in headers for h in ("来源", "版本", "置信度")),
            f"headers={headers}",
        )
        page.screenshot(path=str(OUT_DIR / "uat-hit.png"), full_page=True)
        record("hit_screenshot", True, "uat-hit.png")

        # ── Flow 2: ABSTAIN ──
        page.locator("#symptoms").fill(ABSTAIN_INPUT)
        page.locator("#submit-button").click()
        expect(page.locator("#result")).to_have_attribute(
            "data-status", "ABSTAINED", timeout=30000
        )
        expect(page.locator('.state-message[data-kind="abstained"]')).to_contain_text(
            "检索未命中，已弃权"
        )
        page.screenshot(path=str(OUT_DIR / "uat-abstain.png"), full_page=True)
        record("abstain_state", True, "data-status=ABSTAINED")

        # ── Flow 3: INVALID INPUT (blank) ──
        page.locator("#symptoms").fill("   ")
        page.locator("#submit-button").click()
        expect(page.locator("#result")).to_have_attribute(
            "data-status", "INVALID_INPUT", timeout=30000
        )
        expect(page.locator('.state-message[data-kind="invalid"]')).to_contain_text(
            "输入无效"
        )
        page.screenshot(path=str(OUT_DIR / "uat-invalid.png"), full_page=True)
        record("invalid_state", True, "data-status=INVALID_INPUT")

        browser.close()

    # ── Response-level assertions ──
    statuses = [s for s, _ in diagnoses_responses]
    record(
        "all_post_200",
        all(s == 200 for s in statuses),
        f"statuses={statuses}",
    )
    if len(diagnoses_responses) == 3:
        top_level_sets = [set(body.keys()) for _, body in diagnoses_responses]
        stable = all(s == TOP_LEVEL_FIELDS for s in top_level_sets)
        record("three_state_field_stable", stable, f"fields={sorted(TOP_LEVEL_FIELDS)}")
    else:
        record("three_state_field_stable", False, f"captured={len(diagnoses_responses)}")

    for _, body in diagnoses_responses:
        if body is None:
            continue
        status = body.get("status")
        results["responses"][status] = body
        if status == "DIAGNOSED":
            allowed = {
                node.get("id")
                for node in body.get("verified_knowledge", [])
            }
            refs = []
            diag = body.get("diagnosis") or {}
            refs += diag.get("referenced_entity_ids", []) or []
            for s in body.get("model_suggestions", []) or []:
                refs += s.get("referenced_entity_ids", []) or []
            # All references must be inside the verified knowledge entity id set.
            out_of_graph = [r for r in refs if r not in allowed]
            record(
                "hit_grounded",
                not out_of_graph,
                f"refs={refs} out_of_graph={out_of_graph}",
            )

    with open(OUT_DIR / "uat-summary.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'UAT PASS' if results['ok'] else 'UAT FAIL'}")
    print(f"summary written to {OUT_DIR / 'uat-summary.json'}")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
