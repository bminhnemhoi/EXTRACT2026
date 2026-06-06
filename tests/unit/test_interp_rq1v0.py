"""Unit tests for RQ1-v0 (label-free discriminance) + autointerp helpers.

Pure stdlib — no GPU. Runs under pytest or standalone via
``python tests/unit/test_interp_rq1v0.py``.
"""

from __future__ import annotations

from exact_agent.interp import discriminance as dc
from exact_agent.interp.autointerp import (
    Context,
    build_label_prompt,
    parse_label,
    select_top_contexts,
)
from exact_agent.interp.records import ActivationRecord


def _two_class_samples() -> list[tuple[set[int], str]]:
    a = [({1, 2, 3, i}, "A") for i in range(6)]
    b = [({7, 8, 9, 100 + i}, "B") for i in range(6)]
    return a + b


def test_jaccard_separation_signal() -> None:
    sep = dc.jaccard_separation(_two_class_samples())
    assert sep.within_mean > sep.across_mean
    assert sep.gap > 0.0
    assert sep.auc > 0.9  # disjoint classes -> strong separation


def test_jaccard_separation_no_signal() -> None:
    # identical feature sets across classes -> no separation, auc ~ 0.5
    samples = [({1, 2, 3}, "A"), ({1, 2, 3}, "B"), ({1, 2, 3}, "A")]
    sep = dc.jaccard_separation(samples)
    assert abs(sep.gap) < 1e-9
    assert abs(sep.auc - 0.5) < 1e-9


def test_loo_1nn_accuracy() -> None:
    nn = dc.loo_1nn_accuracy(_two_class_samples())
    assert nn.n == 12
    assert nn.accuracy > nn.majority_baseline
    assert nn.lift > 0.0
    assert abs(nn.majority_baseline - 0.5) < 1e-9


def test_loo_1nn_degenerate() -> None:
    assert dc.loo_1nn_accuracy([({1}, "A")]).n == 1  # < 2 samples -> safe


def test_select_top_contexts_order_and_dedupe() -> None:
    hits = [
        Context("cap energy", 0.2, "s1"),
        Context("cap energy", 0.9, "s2"),  # dup text, higher act
        Context("voltage drop", 0.5, "s3"),
    ]
    top = select_top_contexts(hits, top_n=2)
    assert [c.text for c in top] == ["cap energy", "voltage drop"]  # deduped, by act
    assert top[0].activation == 0.9


def test_build_label_prompt_contains_snippets_and_escape() -> None:
    p = build_label_prompt([Context("capacitor stores energy", 1.0, "s1")])
    assert "capacitor stores energy" in p
    assert "UNINTERPRETABLE" in p


def test_parse_label_variants() -> None:
    assert parse_label("capacitor energy") == "capacitor energy"
    assert parse_label("Label: electric field") == "electric field"
    assert parse_label("- graduation eligibility") == "graduation eligibility"
    assert parse_label('"voltage divider"') == "voltage divider"
    assert parse_label("UNINTERPRETABLE") == ""
    assert parse_label("") == ""


def test_record_roundtrip_with_gold_label() -> None:
    rec = ActivationRecord(
        sample_id="s1", task_type="physics", question="Find E", layer=12,
        active_feature_ids=[1, 2], active_concepts=["capacitor"],
        gold_concepts=["capacitor", "energy"], solver_ok=True,
        solver_meta={"gt_source": "formula"}, gold_label="capacitor_energy",
    )
    back = ActivationRecord.from_json(rec.to_json())
    assert back == rec
    assert back.gold_label == "capacitor_energy"


if __name__ == "__main__":
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
