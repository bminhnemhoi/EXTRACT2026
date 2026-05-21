# 0024 — Day-23 F3 TF-IDF premise selector

Date: 2026-05-21
Status: Accepted (kept ON — P3-positive, P1-neutral)

## Context

Day-4's `premise_selector` used Jaccard overlap on lemma-like surface
forms — cheap, deterministic, zero deps. The official 2026-05-15
release includes a `idx` field per question (1-based premise indices
actually used; CHANGELOG_TYPE1 calls this the "P3 cited-premises
gold"), so we can now *bench* the ranker objectively rather than
trusting the Day-4 heuristic.

## Change

`premise_selector.py`:

* New `_rank_tfidf(question, premises)` — per-query TF-IDF cosine over
  the existing `_normalize` preprocessing (lowercase + stopword strip
  + light stem). `TfidfVectorizer(ngram=(1,1), sublinear_tf=True)`.
* `rank_premises(question, premises, *, method="tfidf")` — TF-IDF is
  the new default. Legacy Jaccard reachable via `method="jaccard"`.
* Defensive fallback: if all texts preprocess to empty (extreme
  stopword case), return zero-score in original order rather than
  letting sklearn raise.

`scripts/bench_premise_selector.py` — new bench: computes Recall@k
against the gold `idx` for both methods on the official 2026-05-15
logic eval split.

## Measured impact

**Bench (Recall@k against gold `idx`, 80 rows with non-empty gold):**

| Method | R@1 | R@3 | R@5 | R@8 |
|---|---:|---:|---:|---:|
| Jaccard (legacy) | 32.6% | 62.4% | 79.4% | 89.6% |
| **TF-IDF (new default)** | **33.1%** | **63.8%** | **81.3%** | 89.0% |
| Δ | +0.5pp | +1.4pp | **+1.9pp** | −0.6pp |

TF-IDF wins at tight k (R@1/R@3/R@5) where it matters for the
``premises`` field of the API response — fewer cited premises but more
of them are the right ones. At R@8 the two converge (both ~89-90%);
that's the per-record cap on what surface similarity can recover
without semantic understanding.

**Logic eval P1 (full pipeline on official 2026-05-15 holdout, 81 rows):**

| | Day-22 (E5 / Jaccard) | **F3 (TF-IDF default)** |
|---|---:|---:|
| Overall correct | 23.5% (19/81) | 23.5% (19/81) |
| MC | 33.3% | 33.3% |
| YNU | 21.2% | 21.2% |
| Abstained | 40.7% | 39.5% |
| ms/sample | 1909 | 2000 |

P1 **unchanged** — the bottleneck is the LLM NL→FOL translator
quality on the 3B backbone (ADR 0018), not premise retrieval. F3
improves the *quality* of the cited evidence (R@5 +1.9pp on gold)
without moving the label-match score.

## Decision — keep ON

Same logic as E5 (ADR 0018) and E8 (ADR 0021): P1-neutral but
P3-positive infrastructure that costs nothing at runtime. The Public
Test Day jury (Slide 12; QA Q21) reviews the cot/premises evidence
directly — having a more precise citation set is a real (small) win
there, and it costs zero latency (~+100ms, well within 60s/req).

If the backbone is ever upgraded (E7 deferred), TF-IDF's tighter
top-k may compound with a stronger NL→FOL translator that uses
fewer-but-better premises in its context window.

## Honest caveat

R@8 dropped 0.6pp — TF-IDF down-weights ubiquitous tokens but a
common-word match that Jaccard kept can occasionally still be a
genuine signal we now miss. Acceptable trade for the tight-k wins.

## Tests

253 passed (no test changes — the existing
`test_premise_selector.py` tests for "well-structured curriculum"
ordering and zero-overlap stability hold under both methods because
the new path preserves the dataclass contract and tie-break rule).
ruff/mypy clean.

## Reproduction

```powershell
uv run python scripts/bench_premise_selector.py
uv run python scripts/run_eval.py --task logic --with-llm `
    --split data/official_v20260515/eval_split/logic_eval.jsonl `
    --out outputs/eval/f3_logic_tfidf
```
