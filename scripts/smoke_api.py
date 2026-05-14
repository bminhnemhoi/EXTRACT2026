"""Send a handful of canonical payloads to a running API and print pass/fail.

Usage:
    uv run python scripts/smoke_api.py [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import httpx

# Force UTF-8 on stdout/stderr so Unicode characters in payloads/responses
# (e.g. μ, →) don't crash on Windows consoles defaulting to cp1252.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class SmokeCase:
    name: str
    payload: dict[str, object]
    expected_task: str


CASES: list[SmokeCase] = [
    SmokeCase(
        name="physics_capacitor_energy",
        payload={
            "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
        },
        expected_task="physics",
    ),
    SmokeCase(
        name="physics_ohm",
        payload={
            "question": (
                "Apply Ohm's law: with R = 5 ohm and I = 2 A, what is the voltage V?"
            )
        },
        expected_task="physics",
    ),
    SmokeCase(
        name="physics_parallel_resistance",
        payload={
            "question": "Two resistors R1 = 4 ohm and R2 = 6 ohm are connected in parallel."
        },
        expected_task="physics",
    ),
    SmokeCase(
        name="logic_simple_modus_ponens",
        payload={
            "premises-NL": [
                "If a student completes all required courses, they are eligible for graduation.",
                "Alice has completed all required courses.",
            ],
            "question": "Is Alice eligible for graduation?",
        },
        expected_task="logic",
    ),
    SmokeCase(
        name="logic_yes_no_unknown",
        payload={
            "premises-NL": ["All birds can fly.", "Penguins are birds."],
            "question": "Can penguins fly?",
        },
        expected_task="logic",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    failures = 0

    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{base}/healthz")
        if health.status_code != 200:
            print(f"[FAIL] healthz status={health.status_code}")
            return 1
        print(f"[OK]   healthz {health.json()}")

        for case in CASES:
            try:
                resp = client.post(f"{base}/predict", json=case.payload)
            except httpx.HTTPError as exc:
                print(f"[FAIL] {case.name} transport error: {exc}")
                failures += 1
                continue

            if resp.status_code != 200:
                print(f"[FAIL] {case.name} status={resp.status_code} body={resp.text}")
                failures += 1
                continue

            body = resp.json()
            ok_required = "answer" in body and "explanation" in body
            ok_task = body.get("task_type") == case.expected_task
            tag = "OK  " if (ok_required and ok_task) else "WARN"
            if not (ok_required and ok_task):
                failures += 1
            print(f"[{tag}] {case.name} -> {json.dumps(body, ensure_ascii=False)[:140]}")

    print()
    print(f"summary: {len(CASES) - failures}/{len(CASES)} cases passed")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
