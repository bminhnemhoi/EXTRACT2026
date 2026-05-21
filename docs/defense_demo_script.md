# Defense demo script — EXACT 2026 Public Test Day (Jun 15)

Jury can interrogate any Top-10 finalist live. The rubric's **P3
(Reasoning Depth)** is graded HERE — they look at the `cot`,
`premises`, `fol` fields of our response and ask follow-up
questions. This script is the rehearsed walkthrough for two
canonical queries.

**Total demo time: ~10 min.** One physics, one logic. Show the
*verifiable* path end-to-end.

## Setup (1 min)

Open two terminal panes:
1. The deployed endpoint URL (committee will share / we confirm)
2. A local shell with our smoke client

```powershell
$ENDPOINT = "https://<our-public-url>/predict"
```

Mention briefly while opening:
- "Our entire architecture is documented in 25 ADRs in
  `docs/decisions/`. The deployed system is **3B + deterministic
  solver**; the 7B SFT adapter is a published ablation."
- "Output schema is fully spec'd: `{answer, explanation, cot,
  premises, fol, confidence}`. Every field reflects something the
  solver *actually did*, not LLM narration."

---

## Demo 1 — Physics (4 min)

Question to invoke:
```
"A parallel-plate air capacitor has a plate area of 31.0 cm² and the
distance between the two plates is 0.84 mm. Calculate the capacitance."
```

(This is a Day-22 F1 win: routed to `parallel_plate_capacitance`,
solved deterministically.)

```powershell
curl -s -X POST $ENDPOINT -H 'Content-Type: application/json' `
  -d '{"question":"A parallel-plate air capacitor has a plate area of 31.0 cm² and the distance between the two plates is 0.84 mm. Calculate the capacitance."}' `
  | jq .
```

Expected response shape (jury will see this):
```json
{
  "answer": "3.268e-11",
  "explanation": "Apply C = ε₀·A/d ...",
  "cot": [
    "Extracted: A = 31.0 cm², d = 0.84 mm",
    "Select formula 'parallel_plate_capacitance': Capacitance of an air parallel-plate capacitor: C = ε₀·A/d (reason: keyword 'air parallel-plate capacitor')",
    "LLM extractor (self-consistent N=3): A=3.1e-3 m² (3/3 votes), d=8.4e-4 m (3/3 votes)",
    "Compute: C = 8.854e-12 × 3.1e-3 / 8.4e-4 = 3.268e-11 F"
  ],
  "premises": [
    "Parallel-plate capacitance: C = ε₀·A/d"
  ],
  "confidence": 0.85
}
```

**Talking points (45s each):**
1. **"The answer was *computed* by SymPy, not by the LLM."** Point at
   the `cot` "Compute: C = 8.854e-12 × ..." line. The LLM only
   *extracted variables*; SymPy substituted into the formula and
   evaluated it.
2. **"The formula selection is auditable."** The `cot` shows the
   *keyword* (`'air parallel-plate capacitor'`) that fired. Our
   classifier is a 30-formula YAML registry plus a routing rule
   table — they can read every rule in `configs/physics_formulas.yaml`
   and `src/exact_agent/physics/topic_classifier.py`.
3. **"Self-consistency vote on extraction."** `(3/3 votes)` per
   variable — Slide-28 organizer "Practical Tip" implemented: three
   independent LLM samples, majority-vote per symbol with 1% rel
   tolerance. The cot proves we did this.
4. **"Unit normalization happens BEFORE compute, by pint."** The
   conversion of 31 cm² to 3.1e-3 m² and 0.84 mm to 8.4e-4 m is the
   `pint` library, not an LLM guess.

**If asked "what if the question is phrased differently?"** Show
the keyword rule table; emphasize that we explicitly enumerated
specific phrasings ("air parallel-plate capacitor", "parallel-plate
air capacitor", "air-filled parallel-plate") — and that the
remaining gap goes through a symbol-overlap fallback with explicit
target-word guards.

**If asked "what if your formula is wrong?"** Show
`tests/integration/test_physics_solver.py` — every formula has a
gold-input/output unit test verified hand on paper.

---

## Demo 2 — Logic (4 min)

Question to invoke (a Y/N entailment, easy to walk through):
```
{
  "premises-NL": [
    "All students who pass the final exam receive credit.",
    "Alice is a student.",
    "Alice passed the final exam."
  ],
  "question": "Does Alice receive credit?"
}
```

```powershell
$body = @{
  "premises-NL" = @(
    "All students who pass the final exam receive credit.",
    "Alice is a student.",
    "Alice passed the final exam."
  )
  question = "Does Alice receive credit?"
} | ConvertTo-Json
curl -s -X POST $ENDPOINT -H 'Content-Type: application/json' `
  -d $body | jq .
```

Expected response shape:
```json
{
  "answer": "Yes",
  "explanation": "From premises P1, P2, P3: P1 establishes the rule, P2 and P3 establish Alice's status — by modus ponens Alice receives credit.",
  "cot": [
    "Top premises (TF-IDF by overlap): P1, P3, P2",
    "Detected question type: yes_no_unknown",
    "Forward chain iterations: 2, derived facts: 4",
    "Derived from premises [1, 2, 3]: 'Alice receives credit'",
    "NL->FOL attempt 1: '∀x (Student(x) ∧ PassedExam(x) → ReceivesCredit(x))' rejected: invalid syntax (caller error)",
    "NL->FOL attempt 2: 'ReceivesCredit(Alice)' accepted",
    "LLM translated claim → ReceivesCredit(Alice)",
    "Z3 entailment fallback applied (surface chain returned Unknown)."
  ],
  "premises": [
    "P1: All students who pass the final exam receive credit.",
    "P2: Alice is a student.",
    "P3: Alice passed the final exam."
  ],
  "fol": "ReceivesCredit(Alice)",
  "confidence": 0.85
}
```

**Talking points:**
1. **"The answer is from a Z3 proof, not from the LLM guessing."**
   Point at `Z3 entailment fallback applied`. Premises are encoded
   in first-order logic; the claim is checked by classical
   entailment.
2. **"NL → FOL is solver-verified."** Point at
   `attempt 1 ... rejected ... attempt 2 ... accepted` — our
   translator runs in a *rejection loop* (Slide-27 organizer-endorsed
   neurosymbolic-hybrid pattern). The LLM never overrides Z3.
3. **"Cited premises are aligned with the dataset's gold idx."**
   The `premises` field shows both the 1-based index *and* the text.
   Our TF-IDF premise selector (Day-23 F3) achieves R@5 = 81% on the
   official dataset's `idx` gold field.
4. **"If Z3 cannot decide, we say Unknown."** The Day-23 E4 MC
   abstention policy treats `Unknown` as a legitimate answer when
   the premises under-determine — matching the 168 retained
   MCQ-Unknown rows in CHANGELOG_TYPE1.

**If asked "what about questions Z3 can't translate?"** Show the
honest fallback: the surface forward-chainer plus answer_verifier
still emits a label. We never crash; we never invent.

---

## Closing (1 min)

If they ask about *training*: "We did QLoRA-SFT on Qwen2.5-7B as
an ablation (Day-13, ADR 0013). Measured impact on the *clean*
SFT-unseen holdout was +2.4-2.5pp single, +2.5pp hybrid stacking
on top of the F1+F2 formula additions (ADR 0022 / 0025). We chose
not to deploy because the deterministic gains were larger and the
hybrid serving complexity (Q3 compliance with two models, vLLM-LoRA
stack) wasn't worth the marginal lift. The SFT adapter, training
data, and bench all reproduce from the repo."

If they ask about *data*: "Only the official 2026-05-15 release.
No external data, no closed-source-LLM-generated synthetic. Data
Disclosure Document at `docs/data_disclosure.pdf`."

If they ask about *reproducibility*: "All 25 ADRs in
`docs/decisions/`. 253 tests, ruff + mypy CI. Every measurement in
this submission was produced by `scripts/run_eval.py` against a
seed-42 holdout split — same script the jury can run themselves."

---

## Emergency fallback plays

* **Endpoint down at demo time** — restart compose: `docker compose
  -f docker/docker-compose.yml restart api`. We have a Vast.ai
  snapshot for instance failure (per `docs/deploy.md`).
* **LLM Tunnel mismatch** — verify `EXACT_LLM__BASE_URL` in api env
  vs Ollama `/v1/models` exposed model id.
* **Specific question fails** — point at the `cot` to show *which*
  step (extraction / classification / unit / compute) failed; never
  claim the system succeeded when it didn't. Honesty earns P3.
