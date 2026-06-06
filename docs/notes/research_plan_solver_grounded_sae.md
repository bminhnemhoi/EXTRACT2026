# Research Charter — Solver-Grounded Agentic Interpretability cho Explainable Educational QA

> **Mục đích của tài liệu:** kim chỉ nam chống-đi-sai-hướng cho hướng nghiên cứu kết hợp Qwen-Scope (SAE) + agentic loop + solver-first. Mọi thí nghiệm/claim phải soi lại §4 (novelty), §5 (invariants & traps), §7 (compliance). Tài liệu này **thay thế** phần mơ hồ của [idea.md](idea.md) (idea.md vẫn giữ làm bản brainstorm gốc).
>
> **Trạng thái:** v1, viết 2026-06-05. Đã neo trên web-research đã kiểm chứng đối kháng (xem §11). Tài liệu **sống** — cập nhật khi chốt decision points (§10) và sau mỗi phase.
>
> **Một dòng (elevator):** Trong một agent QA giáo dục *solver-first*, dùng **symbolic solver làm "người thầy" ground-truth** trong một vòng lặp agentic; SAE (Qwen-Scope) + steering + prompt-revision chỉ là **tay chân (actuator) + kính chẩn đoán (diagnostic)**, **không bao giờ** là người chấm.

---

## 1. Bối cảnh & tài sản sẵn có

- Hệ đang chạy: neuro-symbolic **solver-first** — LLM (hiện Qwen2.5-3B) **không quyết đáp án**; SymPy+pint giải physics, forward-chaining+Z3 giải logic; LLM chỉ trích đại lượng, dịch NL→FOL, diễn đạt explanation từ trace. Scores tham chiếu: physics ~66.9% Full✓ (ADR-0041), logic ~28% Correct (ADR-0036). (Chạy eval để xác nhận, đừng tin doc.)
- Dữ liệu: logic 464 records/913 câu, physics 5,520 bài; holdout chuẩn: `physics_eval_sft_unseen.jsonl` (163) + `logic_eval.jsonl` (81).
- Tài sản then chốt cho hướng này: **solver đã cho ground-truth derivation** (premise/công thức/đơn vị nào được dùng) cho từng câu — đây là thứ cả field SAE đang thiếu (§4).

## 2. Câu hỏi nghiên cứu (RQs)

- **RQ1 (faithfulness có ground-truth):** Feature SAE đang active lúc LLM trích/dịch có **khớp** với khái niệm mà solver xác nhận là đã dùng không, vượt mức ngẫu nhiên? Khi nào SAE *đồng ý* / *mâu thuẫn* với ground-truth?
- **RQ2 (task-focusing):** Đóng/mở (steer) feature off-topic theo loại câu hỏi có **giảm một lớp lỗi đo được** (code-switching, distractor, FOL hỏng) **hơn baseline** (prompting, linear probe) mà **không** làm hỏng coherence không?
- **RQ3 (agentic loop có thầy ground-truth):** Vòng lặp *solver-grounded* (steer + sửa prompt, gate bằng solver) có **hơn** vòng lặp *LLM-critique* (TextGrad/DSPy-style, không solver) và hơn single-pass không? Tỉ lệ "sửa hỏng câu đúng" có ≈ 0 nhờ solver-gate không?

## 3. Đóng góp tuyên bố (claims — EN cho paper)

**Working title:** *VeriScope: A Solver-Grounded Agentic Interpretability Loop for Explainable Educational QA*.

1. **A symbolic-solver-as-ground-truth testbed for SAE faithfulness.** We use a neuro-symbolic QA agent whose deterministic solver yields a faithful derivation as ground truth, and measure whether the LLM's SAE features at extraction/translation time align with solver-confirmed task concepts — addressing the field's well-documented "absence of ground truth" for SAE evaluation.
2. **A solver-grounded agentic loop (offline-discover → deploy-static).** A controller performs question-conditioned activation steering and solver-feedback-driven prompt revision; the symbolic solver is the in-loop judge (not the LLM, not the SAE), making the loop monotone (never ships a revision unless the solver score improves).
3. **An honest empirical finding** comparing the solver-grounded loop against LLM-critique loops and prompting, plus a characterization of where representation-level (SAE) diagnosis agrees/disagrees with the symbolic ground truth — a positive *or* negative result, both publishable in a skeptical field.

> ⚠️ **Đóng góp KHÔNG phải** là "dùng SAE/steering/prompt-opt" (đều là prior art — §4). Đóng góp là **solver làm thầy ground-truth trong loop** + **đánh giá faithfulness có ground truth**.

## 4. Định vị so với prior art — cái gì MỚI, cái gì ĐÃ CÓ (phần chống-đi-sai-hướng cốt lõi)

| Thành phần | Đã tồn tại (KHÔNG claim là mới) | Ref |
|---|---|---|
| Bộ SAE mở để soi model | Gemma Scope (DeepMind, 2024) — Qwen-Scope = "Gemma Scope cho Qwen" | 2408.05147 / 2605.11887 |
| Truy vết "đáp án đến từ đâu" | Attribution graphs / circuit tracing (Anthropic 2025) — mạnh hơn SAE, đã open-source | transformer-circuits 2025 |
| Đóng/mở feature theo input | Weighted/Adaptive Activation Steering, Control-RL | 2505.20309 / 2406.00034 / 2602.10437 |
| Sửa prompt qua feedback loop | ProTeGi, TextGrad, OPRO, DSPy, SIPDO, CriSPO | 2502.16923 (survey) |
| Gán nhãn NL cho feature | AutoInterp | 2410.13928 |
| Steer giảm off-topic | off-topic latents, topic alignment, EnSToM | 2602.06941 / 2506.12576 |

**Khoảng trống MỚI ta lấp:** mọi loop ở trên dùng judge = LLM-critique / synthetic data / textual gradient (đều noisy); và mọi SAE-eval thiếu ground truth (phải dùng toy/supervised probe — SAGE 2410.07456, 2405.08366). **Ta thay judge bằng solver xác định + dùng solver trace làm ground truth.** Đó là điểm khác biệt duy nhất cần bảo vệ.

## 5. Bất biến & các bẫy phải tránh (đọc trước mỗi thí nghiệm)

**Invariants (không được phá):**
1. **Solver = người chấm.** SAE và LLM **không bao giờ** quyết đúng/sai hay quyết đáp án.
2. **Solver-first:** LLM chỉ trích/dịch/diễn đạt; đáp án do SymPy/pint + Z3 quyết.
3. **Monotone:** chỉ giữ bản sửa khi điểm solver tăng (tái dùng gate trong [self_correction.py](../../src/exact_agent/agent/self_correction.py)).
4. **Có ground truth mới đo faithfulness.** Không suy diễn nhân quả từ activation đơn thuần.

**Anti-patterns (làm là chết):**
- ❌ Để **SAE làm judge** off-track → chồng SAE-không-tin-cậy lên self-correction-degrade → loop đuổi nhiễu, kéo accuracy xuống (Huang 2023 + 2601.05679).
- ❌ Claim **"SAE tiết lộ suy luận thật"** → bị bác (2601.05679; faithfulness của attribution là việc của solver trace).
- ❌ Framing **"combine 3 kỹ thuật"** mà không có solver-ground-truth + finding → reviewer chê incremental.
- ❌ Không có **baseline** (prompting, linear probe, LLM-critique loop) → AxBench-precedent bắt buộc phải có.
- ❌ Steer **α cao** → sụp coherence (Rogue Scalpel 2509.22067); luôn có α-sweep + kiểm coherence.

## 6. Kiến trúc (phân vai rõ)

```
Câu hỏi
  ↓
[Controller] route logic/physics  → chọn cấu hình steering theo loại câu (offline đã tối ưu)
  ↓
LLM (Instruct) trích đại lượng / dịch NL→FOL   ← (tùy chọn) steering dập feature off-topic
  ↓
SOLVER (SymPy/pint | forward-chaining/Z3)  ←—— NGƯỜI THẦY: quyết đúng/sai + lý do có cấu trúc
  ↓ (nếu solver báo sai)
[Prompt reviser] dùng LÝ DO của solver (+ gợi ý kiểu-lạc-hướng từ SAE) để viết prompt "rào trước"
  ↓  lặp lại, GATE bằng điểm solver (monotone, ≤2–3 vòng)
Explanation từ solver trace (faithful)  +  SAE feature report (diagnostic, đã validate vs solver)
  ↓
JSON: answer/explanation/cot/premises/fol/confidence
```

**Reframe triển khai (quan trọng):** chạy loop **offline lúc dev** để *khám phá* `(prompt tốt + cấu hình steering)` cho mỗi loại câu; **deploy bản tĩnh đã chưng cất** (prompt cố định + vector steering cố định). → live system vẫn solver-first, nhanh, không rủi ro runtime; đúng cách DSPy/OPRO được dùng.

- **Phân tích SAE chạy trên bản Base** (SAE Qwen-Scope là cho Base); agent chạy Instruct → tách 1 lượt forward Base chỉ để lấy activation (xem §10 decision).

## 7. Ràng buộc tuân thủ (vi phạm = hỏng/loại)

- **LLM ≤8B open-source.** Qwen3-8B = 8.2B *tổng* / 6.95B *non-embedding* → biên; xác nhận convention đếm tham số, nếu không chắc dùng **Qwen3.5-2B** (sạch luật) (§10).
- **Mọi dataset fine-tune phải công khai** (nếu còn dự thi). Phase hiện tại **không cần training** → tránh được rủi ro này; nếu sau dùng SASFT/GRPO thì phải release data.
- **License SAE "qwen"** (không phải Apache-2.0) — xác nhận được phép dùng artifact non-OSI. (Gemma Scope là CC-BY-4.0 nhưng Gemma2-9B>8B; chỉ 2B lọt — phương án dự phòng.)
- **`answer`+`explanation` bắt buộc**; `fol/cot/premises/confidence` tùy chọn (tăng P3).

## 8. Thiết kế thí nghiệm

**Models:** chính = Qwen3-8B (Base cho SAE, Instruct cho agent); dự phòng eligibility = Qwen3.5-2B. So với baseline Qwen2.5-3B hiện tại.

**Splits:** dùng holdout chuẩn (physics 163 SFT-unseen, logic 81); không thay holdout.

**Metrics:**
- *P1/P2/P3* theo eval harness ([eval/metrics.py](../../src/exact_agent/eval/metrics.py)): physics Full✓ (numeric∧unit), logic label-match; P2/P3 qua proxy + review.
- *RQ1 — SAE alignment:* precision/recall giữa top-active features (lúc trích/dịch) và khái niệm solver-confirmed, so **random baseline** + so **linear probe**.
- *RQ2 — steering ablation:* tỉ lệ code-switching, lỗi distractor, FOL-malformed **trước/sau** steer; α-sweep; kiểm coherence (không tụt).
- *RQ3 — loop:* P1/P2 lift của {single-pass | LLM-critique loop | solver-grounded loop}; **over-correction rate** (đúng→sai) — phải ≈0 ở solver-grounded.

**Baselines bắt buộc:** prompting thường; linear/diff-in-means probe (thay SAE); LLM-critique prompt-opt (TextGrad/DSPy) *không* solver.

**Kill / success criteria (quyết định hướng có chân):**
- ✅ Đi tiếp nếu: RQ1 alignment > random *có ý nghĩa*; RQ3 solver-loop > single-pass trên P1/P2 với over-correction≈0.
- ⛔ Pivot nếu: RQ1 ≈ random **và** RQ3 không hơn baseline → khi đó **negative result trung thực** (vẫn viết được, đúng tinh thần field), và đòn deploy quay về *explanation bám solver-trace giàu hơn* (vẫn ăn P2/P3).

## 9. Lộ trình theo phase (mỗi phase = 1 deliverable, theo kỷ luật ADR)

- **P0 — Setup & decisions:** chốt §10; dựng môi trường Python 3.11; tải model + SAE (sae_lens/loader Qwen); kiểm license. *Deliverable:* ADR chọn model + ghi chú license.
- **P1 — Harness bắt activation:** hook residual stream ở layer SAE, lưu activation + feature code + output + solver verdict theo từng dòng (đọc [orchestrator.py](../../src/exact_agent/agent/orchestrator.py), [vllm_client.py](../../src/exact_agent/llm/vllm_client.py) để cắm hook). *Deliverable:* dataset activation + script tái lập.
- **P2 — RQ1 alignment study:** đo khớp feature↔ground-truth + figure. *Deliverable:* bảng alignment vs random/probe.
- **P3 — RQ2 steering ablation:** dập off-topic/code-switch, α-sweep, đo lớp lỗi + coherence. *Deliverable:* bảng ablation + baseline.
- **P4 — RQ3 agentic loop (offline-discover):** loop solver-gated; so 3 cấu hình; đo over-correction. *Deliverable:* bảng so loop + prompt/steering tĩnh đã chưng cất.
- **P5 — Deploy-static + write-up:** nhúng prompt+steering tĩnh; eval cuối; viết paper (related work §4 + design + limitations). *Deliverable:* short-paper draft + reproducibility.

## 10. Decision points cần bạn chốt (đặt default — không chặn việc viết)

1. **Model:** ✅ **CHỐT 2026-06-06 — giữ Qwen-Scope / Qwen3-8B-Base** (đã implement + verify; dev trên Qwen3.5-2B-Base, headline Qwen3-8B-Base). Đã cân nhắc **Gemma Scope 2 / Gemma 3 4B-it** (instruct-SAE + transcoder/cross-layer = circuit-tracing + Matryoshka SAE + sae_lens, 4B ≤8B sạch) — mạnh hơn về *giá trị* nhưng tốn re-tool (đổi base + loader). **Để dành làm hướng nâng cấp / cross-suite comparison** nếu RQ1 trên Qwen yếu hoặc muốn provenance cấp-mạch. Vẫn nên hỏi BTC convention đếm tham số (8.2B biên); nếu nghiêm → submit bằng Qwen3.5-2B.
2. **Base vs Instruct cho phân tích:** *default* agent chạy Instruct + lượt Base riêng để lấy activation (SAE exact). Phương án rẻ hơn: chấp nhận SAE-Base-trên-Instruct (xấp xỉ, phải nói rõ).
3. **Đích nộp:** *default* special-session XAII của chính EXACT (rất khớp) + bản nộp cuộc thi. (Top-conference: không đặt kỳ vọng.)

## 11. Reading list đã khóa (nhóm theo vai — trích dẫn từ đây, đừng tự bịa)

- **Công cụ SAE:** Qwen-Scope 2605.11887 · Gemma Scope 2408.05147 · AutoInterp 2410.13928 · Anthropic attribution graphs (transformer-circuits 2025: methods + "biology").
- **Giới hạn SAE (để phòng-thủ reviewer):** AxBench 2501.17148 · DeepMind "negative results / deprioritising SAE" (2025) · Sanity Checks 2602.14111 · Falsifying SAE reasoning features 2601.05679 · "discover not act" 2506.23845 · "good for steering if you select features" 2505.20063.
- **Ground-truth SAE eval (khoảng trống ta lấp):** SAGE 2410.07456 · Principled Eval 2405.08366 · CE-Bench 2509.00691.
- **Steering controller / adaptive / off-topic:** WAS 2505.20309 · Adaptive Activation Steering 2406.00034 · Control-RL 2602.10437 · conditional SAE clamping 2503.11127 · EnSToM 2505.16526 · off-topic latents 2602.06941 · topic alignment 2506.12576 · Rogue Scalpel 2509.22067.
- **Prompt-opt loop:** ProTeGi · TextGrad (Yuksekgonul 2024) · OPRO · DSPy · SIPDO 2505.19514 · CriSPO · survey 2502.16923.
- **Self-correction & verifier (lý do solver làm thầy):** Huang 2310.01798 · CRITIC 2305.11738 · Self-Refine 2303.17651 · Reflexion 2303.11366 · PRM/Lightman 2305.20050 · Math-Shepherd 2312.08935 · RLVR/DeepSeek-R1 2501.12948 · GRPO 2402.03300.
- **Faithfulness / faithful-CoT:** Faithful-CoT 2301.13379 · SymbCoT 2405.18357 · VeriCoT 2511.04662 · Turpin 2305.04388 · Chen/Anthropic 2505.05410 · Lanham 2307.13702.

> Số liệu cần sửa khi trích: DeepSeek-R1-Zero AIME = **71.0% pass@1** (không phải 77.9%); **không** dùng Lanham để nói "model nhỏ = kém faithful" (Lanham là inverse-scaling) — motivate bằng Turpin + Chen/Anthropic.

## 12. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Không có SAE cho Qwen2.5-3B | Đổi base sang model có SAE (§10); deploy-static để không cần SAE lúc chạy live |
| SAE faithfulness yếu (RQ1 ≈ random) | Đã có kill-criteria → pivot sang negative-result + explanation-trace |
| Loop làm tệ đi | Solver-gate monotone; over-correction≈0 là điều kiện ship |
| Steering hỏng coherence | α-sweep + coherence guard; ưu tiên feature ngôn ngữ (sạch nhất) |
| Latency | Offline-discover → deploy-static |
| Eligibility 8.2B | Fallback Qwen3.5-2B; hỏi convention BTC |
| Dataset disclosure | Phase hiện tại không train; nếu train thì release |
```
