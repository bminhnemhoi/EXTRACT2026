"""Unit tests for the pure interpretability modules (no GPU/torch needed).

Covers the RQ1 metric (alignment), concept derivation, and record I/O. The
GPU-gated modules (sae_loader/hooks/steering) are intentionally not exercised
here. Runs under pytest, or standalone via ``python tests/unit/test_interp_alignment.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from exact_agent.interp import alignment as al
from exact_agent.interp.concepts import (
    split_identifier,
    tokenize_concepts,
    topic_concepts,
)
from exact_agent.interp.features import (
    concepts_from_features,
    load_label_cache,
    save_label_cache,
    top_k_indices,
)
from exact_agent.interp.ground_truth import (
    extract_predicates,
    logic_concepts,
    physics_concepts,
)
from exact_agent.interp.records import ActivationRecord


def test_prf_basic() -> None:
    p = al.prf({"a", "b"}, {"a", "b", "c"})
    assert p.n_overlap == 2
    assert p.precision == 1.0
    assert abs(p.recall - 2 / 3) < 1e-9
    assert abs(p.f1 - (2 * 1.0 * (2 / 3)) / (1.0 + 2 / 3)) < 1e-9


def test_prf_empty_boundaries() -> None:
    assert al.prf(set(), set()).f1 == 1.0
    assert al.prf({"x"}, set()).precision == 0.0
    assert al.prf(set(), {"x"}).recall == 0.0


def test_aggregate() -> None:
    rs = [al.prf({"a"}, {"a"}), al.prf({"b"}, {"c"})]
    agg = al.aggregate(rs)
    assert agg.n == 2
    assert abs(agg.macro_f1 - 0.5) < 1e-9  # 1.0 and 0.0
    # micro: overlap 1, active 2, gold 2 -> p=r=0.5 -> f1=0.5
    assert abs(agg.micro_f1 - 0.5) < 1e-9


def test_random_baseline_is_low_for_large_vocab() -> None:
    vocab = [f"c{i}" for i in range(100)]
    gold = {"c0", "c1", "c2"}
    rb = al.random_baseline_f1(vocab, gold, k=3, n_trials=200, seed=1)
    assert rb < 0.2  # random draws rarely hit the 3 gold concepts out of 100


def test_compare_aligned_beats_noise() -> None:
    aligned = ({"capacitor", "voltage"}, {"capacitor", "voltage", "energy"}, 2)
    noise = ({"poetry", "novel"}, {"resistance", "current"}, 2)
    vocab = ["capacitor", "voltage", "energy", "resistance", "current",
             "poetry", "novel", "ohm", "charge", "field"]
    cmp = al.compare([aligned, noise], vocab, n_trials=100, seed=0)
    assert cmp.n == 2
    assert cmp.observed_macro_f1 > cmp.random_macro_f1  # signal beats random
    assert 0.0 <= cmp.win_rate <= 1.0


def test_tokenize_and_identifiers() -> None:
    assert "capacitor" in tokenize_concepts("Energy stored in a capacitor: E = 0.5 C U^2")
    assert "the" not in tokenize_concepts("the of and capacitor")
    assert split_identifier("eligible_for_graduation") == {"eligible", "graduation"}
    assert "circuit" in topic_concepts("TD")


def test_physics_concepts() -> None:
    c = physics_concepts(
        description="Energy stored in a capacitor: E = 0.5*C*U**2",
        input_aliases=["capacitance", "voltage", "V"],
        output_symbol="E",
        topic="TD",
    )
    assert {"capacitor", "energy", "capacitance", "voltage"} <= c
    assert "circuit" in c  # topic expansion


def test_logic_concepts_and_predicates() -> None:
    preds = extract_predicates(["eligible_for_graduation(John)", "GpaAbove(John)"])
    assert "eligible" in preds and "graduation" in preds
    c = logic_concepts(
        supporting_premise_texts=["John completed the required courses."],
        fol_strings=["graduates_with_honors(John)"],
    )
    assert "completed" in c and "honors" in c


def test_top_k_indices() -> None:
    assert top_k_indices([0.1, 0.9, 0.5, 0.95], 2) == [3, 1]
    assert top_k_indices([], 3) == []
    assert top_k_indices([1.0], 5) == [0]


def test_concepts_from_features() -> None:
    labels = {1: "capacitor and energy storage", 2: "Chinese language tokens"}
    got = concepts_from_features([1, 2, 999], labels)
    assert "capacitor" in got and "energy" in got
    assert "chinese" in got and "language" in got  # fid 999 unlabeled -> ignored


def test_label_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "labels.json"
        save_label_cache({1: "capacitor energy", 2: "voltage"}, p)
        loaded = load_label_cache(p)
        assert loaded == {1: "capacitor energy", 2: "voltage"}
    assert load_label_cache(Path(d) / "missing.json") == {}


def test_record_roundtrip() -> None:
    rec = ActivationRecord(
        sample_id="s1", task_type="physics", question="Find E", layer=18,
        active_feature_ids=[1, 2], active_concepts=["capacitor"],
        gold_concepts=["capacitor", "energy"], solver_ok=True,
        solver_meta={"gt_source": "formula"},
    )
    back = ActivationRecord.from_json(rec.to_json())
    assert back == rec


if __name__ == "__main__":  # standalone runner (no pytest needed)
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
