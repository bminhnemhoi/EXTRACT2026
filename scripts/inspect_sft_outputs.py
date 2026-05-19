"""Qualitatively inspect a served SFT model on REAL held-out rows.

Sends the exact (system, user) prompt the model was trained on
(`data/processed/sft_chat_val.jsonl` — held out, never trained) to a
running OpenAI-compatible endpoint, prints the **raw** completion
verbatim, and flags: JSON parseable? braces/quotes balanced? prose
rambling around the JSON? cot/premises format? hallucinated premise
refs? — so you can eyeball quality before trusting the aggregate eval.

Usage:
    uv run python scripts/inspect_sft_outputs.py --url https://x.trycloudflare.com/v1 --n 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from exact_agent.llm.vllm_client import parse_json_completion  # noqa: E402

VAL_PATH = REPO_ROOT / "data" / "processed" / "sft_chat_val.jsonl"
_PREMISE_REF = re.compile(r"\bP\s*?(\d{1,3})\b|\bpremise\s+(\d{1,3})\b", re.IGNORECASE)


def _stratified(rows: list[dict], n: int) -> list[dict]:
    logic = [r for r in rows if r.get("task_type") == "logic"]
    phys = [r for r in rows if r.get("task_type") == "physics"]
    half = max(1, n // 2)
    picked = logic[:half] + phys[: n - len(logic[:half])]
    return picked[:n]


def _premise_count(user_content: str) -> int:
    try:
        payload = json.loads(user_content)
    except json.JSONDecodeError:
        return 0
    return len(payload.get("premises-NL") or payload.get("premises_NL") or [])


def _flags(raw: str, task: str, n_premises: int) -> dict[str, object]:
    text = raw.strip()
    first, last = text.find("{"), text.rfind("}")
    prose_before = text[:first].strip() if first != -1 else text
    prose_after = text[last + 1 :].strip() if last != -1 else ""
    json_ok = True
    obj: object = None
    try:
        obj = parse_json_completion(raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        json_ok = False

    keys = sorted(obj.keys()) if isinstance(obj, dict) else []
    halluc = ""
    if json_ok and isinstance(obj, dict):
        # Logic: any referenced premise index beyond what was given.
        refs: list[int] = []
        blob = f"{obj.get('explanation', '')} {obj.get('premises', '')}"
        for m in _PREMISE_REF.finditer(blob):
            refs.append(int(m.group(1) or m.group(2)))
        bad = [i for i in refs if n_premises and (i < 1 or i > n_premises)]
        if bad:
            halluc = f"premise refs out of range {bad} (have {n_premises})"
        if task == "physics" and not str(obj.get("answer", "")).strip():
            halluc = (halluc + "; " if halluc else "") + "empty physics answer"

    return {
        "json_ok": json_ok,
        "braces_balanced": text.count("{") == text.count("}") and text.count("{") > 0,
        "quotes_even": text.count('"') % 2 == 0,
        "prose_before": prose_before[:80],
        "prose_after": prose_after[:80],
        "char_len": len(text),
        "keys": keys,
        "has_answer": isinstance(obj, dict) and bool(str(obj.get("answer", "")).strip()),
        "has_explanation": isinstance(obj, dict) and bool(str(obj.get("explanation", "")).strip()),
        "hallucination": halluc,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url", required=True, help="endpoint base, e.g. https://x.trycloudflare.com/v1"
    )
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--show-chars", type=int, default=900)
    args = ap.parse_args()

    if not VAL_PATH.exists():
        print(f"[fatal] {VAL_PATH} missing — run prepare_sft first", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in VAL_PATH.open(encoding="utf-8") if line.strip()]
    sample = _stratified(rows, args.n)
    base = args.url.rstrip("/")

    tally = {
        "json_ok": 0,
        "balanced": 0,
        "clean_no_prose": 0,
        "has_answer_expl": 0,
        "halluc": 0,
    }
    with httpx.Client(timeout=120.0) as cli:
        for i, r in enumerate(sample, 1):
            sysm, userm, goldm = r["messages"]
            n_prem = _premise_count(userm["content"])
            try:
                resp = cli.post(
                    f"{base}/chat/completions",
                    json={
                        "model": "exact-sft",
                        "messages": [sysm, userm],
                        "max_tokens": args.max_tokens,
                        "temperature": 0.2,
                        "top_p": 0.9,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                print(f"\n[{i}] {r['sample_id']} ({r['task_type']}) — REQUEST FAILED: {exc}")
                continue

            f = _flags(raw, r["task_type"], n_prem)
            tally["json_ok"] += f["json_ok"]
            tally["balanced"] += f["braces_balanced"] and f["quotes_even"]
            tally["clean_no_prose"] += not f["prose_before"] and not f["prose_after"]
            tally["has_answer_expl"] += f["has_answer"] and f["has_explanation"]
            tally["halluc"] += bool(f["hallucination"])

            try:
                gold_ans = json.loads(goldm["content"]).get("answer", "?")
            except json.JSONDecodeError:
                gold_ans = "?"

            print("\n" + "=" * 78)
            print(f"[{i}] {r['sample_id']}  task={r['task_type']}  premises={n_prem}")
            print(
                f"  flags: json_ok={f['json_ok']} balanced={f['braces_balanced']}/"
                f"{f['quotes_even']} len={f['char_len']} keys={f['keys']} "
                f"answer={f['has_answer']} expl={f['has_explanation']}"
            )
            if f["prose_before"]:
                print(f"  PROSE BEFORE JSON: {f['prose_before']!r}")
            if f["prose_after"]:
                print(f"  PROSE AFTER JSON:  {f['prose_after']!r}")
            if f["hallucination"]:
                print(f"  HALLUCINATION: {f['hallucination']}")
            print(f"  gold answer: {gold_ans!r}")
            print("  --- RAW MODEL OUTPUT ---")
            print(raw[: args.show_chars] + ("…[truncated]" if len(raw) > args.show_chars else ""))

    n = len(sample)
    print("\n" + "#" * 78)
    print(f"SUMMARY over {n} held-out cases:")
    print(f"  valid JSON parse : {tally['json_ok']}/{n}")
    print(f"  braces+quotes ok : {tally['balanced']}/{n}")
    print(f"  no prose rambling: {tally['clean_no_prose']}/{n}")
    print(f"  answer+expl both : {tally['has_answer_expl']}/{n}")
    print(f"  hallucination flag: {tally['halluc']}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
