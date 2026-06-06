"""RQ1-v0 — label-free: do active SAE features discriminate the solver class?

No autointerp labels needed. The only inputs are each sample's active feature
ids (from capture) and its solver ``gold_label`` (physics: formula_id; logic:
answer). If the SAE features carry the task structure the solver uses, then:

* same-class samples share more active features than different-class ones
  (:func:`jaccard_separation`), and
* a sample's class is predictable from its feature set
  (:func:`loo_1nn_accuracy`, leave-one-out 1-NN by feature overlap).

Both are pure stdlib (O(n²) over a few hundred samples — fine) and beat a
random/majority baseline only if there is real signal. This is the fastest
way to get an honest RQ1 read before building the autointerp label layer.
"""

from __future__ import annotations

from dataclasses import dataclass


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


@dataclass(frozen=True)
class Separation:
    n_pairs: int
    within_mean: float
    across_mean: float
    gap: float  # within - across
    auc: float  # P(within-pair overlap > across-pair overlap)


def jaccard_separation(samples: list[tuple[set[int], str]]) -> Separation:
    """Mean feature-set Jaccard for same-class vs different-class pairs.

    ``samples`` = list of ``(active_feature_ids, gold_label)``. AUC is the
    fraction of (within, across) pair comparisons where the within-class pair
    has the larger overlap (0.5 = no signal), estimated over all pair pairings
    via the rank/Mann–Whitney identity.
    """
    within: list[float] = []
    across: list[float] = []
    n = len(samples)
    for i in range(n):
        fi, li = samples[i]
        for j in range(i + 1, n):
            fj, lj = samples[j]
            s = _jaccard(fi, fj)
            (within if li == lj else across).append(s)
    within_mean = sum(within) / len(within) if within else 0.0
    across_mean = sum(across) / len(across) if across else 0.0
    auc = _auc(within, across)
    return Separation(len(within) + len(across), within_mean, across_mean,
                      within_mean - across_mean, auc)


def _auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg) with ties = 0.5 (Mann–Whitney U / |pos||neg|)."""
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


@dataclass(frozen=True)
class NNResult:
    n: int
    accuracy: float
    majority_baseline: float
    lift: float


def loo_1nn_accuracy(samples: list[tuple[set[int], str]]) -> NNResult:
    """Leave-one-out 1-NN classification of gold_label by feature overlap.

    For each sample, predict the gold_label of the single other sample whose
    active-feature set has the highest Jaccard overlap. Compare to the
    majority-class baseline.
    """
    n = len(samples)
    if n < 2:
        return NNResult(n, 0.0, 0.0, 0.0)
    labels = [lab for _, lab in samples]
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    majority = max(counts.values()) / n

    correct = 0
    for i in range(n):
        fi, li = samples[i]
        best_j, best_s = -1, -1.0
        for j in range(n):
            if j == i:
                continue
            s = _jaccard(fi, samples[j][0])
            if s > best_s:
                best_s, best_j = s, j
        if best_j != -1 and labels[best_j] == li:
            correct += 1
    acc = correct / n
    return NNResult(n, acc, majority, acc - majority)
