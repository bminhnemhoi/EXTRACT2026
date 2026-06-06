"""RQ1 metric — does SAE-feature activation align with the solver ground truth?

Pure standard library (no numpy/torch) so it is fully unit-testable and runs
on any Python. Everything is framed as set overlap between two concept sets:

* ``active``  — concepts the SAE features (that fired) map to for a sample.
* ``gold``    — concepts the symbolic solver confirms were used.

The headline question (charter §2, RQ1) is whether the *observed* alignment
exceeds a **random baseline** (drawing the same number of concepts uniformly
from the vocabulary). A linear-probe baseline is computed the same way from
probe-derived concept sets and compared with :func:`compare`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    n_active: int
    n_gold: int
    n_overlap: int


def prf(active: set[str], gold: set[str]) -> PRF:
    """Precision/recall/F1 of ``active`` against ``gold``.

    Conventions at the empty-set boundaries (chosen so aggregates stay sane):
    * no active and no gold  -> p=r=f1=1.0 (vacuously aligned)
    * active but no gold     -> p=0.0, r=1.0
    * gold but no active     -> p=1.0, r=0.0
    """
    overlap = active & gold
    n_a, n_g, n_o = len(active), len(gold), len(overlap)
    if n_a == 0 and n_g == 0:
        return PRF(1.0, 1.0, 1.0, 0, 0, 0)
    precision = (n_o / n_a) if n_a else 0.0
    recall = (n_o / n_g) if n_g else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, n_a, n_g, n_o)


@dataclass(frozen=True)
class Aggregate:
    n: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float


def aggregate(results: list[PRF]) -> Aggregate:
    """Macro (mean of per-sample) and micro (pooled counts) averages."""
    n = len(results)
    if n == 0:
        return Aggregate(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    macro_p = sum(r.precision for r in results) / n
    macro_r = sum(r.recall for r in results) / n
    macro_f1 = sum(r.f1 for r in results) / n
    tot_a = sum(r.n_active for r in results)
    tot_g = sum(r.n_gold for r in results)
    tot_o = sum(r.n_overlap for r in results)
    micro_p = (tot_o / tot_a) if tot_a else 0.0
    micro_r = (tot_o / tot_g) if tot_g else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )
    return Aggregate(n, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1)


def random_baseline_f1(
    vocab: list[str],
    gold: set[str],
    k: int,
    *,
    n_trials: int = 200,
    seed: int = 0,
) -> float:
    """Mean F1 if we drew ``k`` concepts uniformly at random from ``vocab``.

    ``k`` should be the number of distinct concepts the real SAE features
    mapped to for the sample, so the baseline controls for set size.
    """
    if not vocab or k <= 0:
        return prf(set(), gold).f1
    rng = random.Random(seed)
    k = min(k, len(vocab))
    total = 0.0
    for _ in range(n_trials):
        draw = set(rng.sample(vocab, k))
        total += prf(draw, gold).f1
    return total / n_trials


@dataclass(frozen=True)
class Comparison:
    """RQ1 verdict: observed alignment vs random baseline (per-sample mean)."""

    n: int
    observed_macro_f1: float
    random_macro_f1: float
    lift: float  # observed - random
    win_rate: float  # fraction of samples where observed_f1 > its random baseline


def compare(
    samples: list[tuple[set[str], set[str], int]],
    vocab: list[str],
    *,
    n_trials: int = 200,
    seed: int = 0,
) -> Comparison:
    """Compare observed alignment against the random baseline across samples.

    ``samples`` = list of ``(active_concepts, gold_concepts, k)`` where ``k``
    is the size to use for that sample's random draw (usually
    ``len(active_concepts)``).
    """
    obs: list[PRF] = []
    rand_f1s: list[float] = []
    wins = 0
    for i, (active, gold, k) in enumerate(samples):
        p = prf(active, gold)
        obs.append(p)
        rb = random_baseline_f1(vocab, gold, k, n_trials=n_trials, seed=seed + i)
        rand_f1s.append(rb)
        if p.f1 > rb:
            wins += 1
    agg = aggregate(obs)
    rand_macro = (sum(rand_f1s) / len(rand_f1s)) if rand_f1s else 0.0
    win_rate = (wins / len(samples)) if samples else 0.0
    return Comparison(
        n=len(samples),
        observed_macro_f1=agg.macro_f1,
        random_macro_f1=rand_macro,
        lift=agg.macro_f1 - rand_macro,
        win_rate=win_rate,
    )
