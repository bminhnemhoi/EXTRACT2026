# EXTRACT2026 — Phân tích Tiến độ & Kết quả

**Cập nhật:** 2026-05-31 · **Deadline finals:** 2026-06-15 (theo debai.md) · **Status:** Functional end-to-end ✅

> ⚠️ Tài liệu này được dựng lại từ việc **đọc trực tiếp source code** (src/, 41 ADRs, configs, tests, docs) — không chỉ từ README. Mọi con số đều có nguồn. Những chỗ README/HANDOVER bị **lỗi thời** đã được đánh dấu rõ.

---

## 1. Dự án là gì?

**Solver-first neuro-symbolic QA agent** cho cuộc thi **IEEE IJCNN 2026 EXACT** (Explainable Educational QA). Hai loại bài:

- **Logic** — suy luận FOL trên quy định đại học (official: 808 records). Dạng: MC, Yes/No/Unknown, open.
- **Physics** — bài toán số mạch điện + tĩnh điện (official: 1,352 problems).

**Nguyên tắc bất biến (core invariant):** LLM **không bao giờ** quyết định đáp án.
- **Physics** → SymPy + pint tính toán
- **Logic** → forward-chaining + Z3 chứng minh
- **LLM** (Qwen2.5-3B-Instruct via Ollama) chỉ: (a) trích xuất đại lượng, (b) dịch NL→FOL, (c) diễn đạt giải thích từ solver trace.

Submission: `POST /predict` → `{answer, explanation, cot, premises, fol, confidence}`.

---

## 2. Kết quả hiện tại

| Task | Metric | Score | Nguồn |
| --- | --- | --- | --- |
| **Physics** | Full✓ (value+unit, strict) | **~63%** (README/HANDOVER) · **66.9%** (ADR-0041, mới nhất) | README:37, ADR-0041 |
| **Logic** | Correct (exact Y/N/Unknown) | **~24%** (24.7% ADR-0035 → 28.4% ADR-0036) | ADR-0035/36 |
| **Tests** | unit + integration + e2e | **295 functions / 31 files** (README ghi 271 — đã lỗi thời) | đếm thực tế |
| **Formulas** | physics_formulas.yaml | **70 công thức** (README ghi 47 — đã lỗi thời) | đếm thực tế |
| **ADRs** | docs/decisions/ | **41** ADRs | đếm thực tế |

> 🔎 **Lưu ý quan trọng:** thư mục `outputs/eval/` **chưa tồn tại** trong repo — các con số headline nằm trong tài liệu (README/HANDOVER/ADR), **không** phải từ một lần chạy eval còn lưu trên đĩa. Việc đầu tiên nên làm: **chạy lại eval để có số liệu tươi** (xem §7).
>
> 🔎 **Mâu thuẫn nội bộ cần chốt:** README/HANDOVER nói Day-29 = ~63%, nhưng ADR-0041 (cũng Day-29 Iter-19) ghi **66.9%** (60.7%→66.9%, +6.2pp, 10 rows). Cần chạy eval để xác nhận con số đúng trước khi nộp.

---

## 3. Đã làm những gì? (41 ADRs — quỹ đạo đo được)

### Quỹ đạo Physics Full✓ (trên 163-row SFT-unseen holdout)
```
Day-1  (ADR-0003):  0.8%   rule-only baseline
Day-8  (ADR-0008): ~8%    Ollama qwen2.5:3b fallback bật
Day-12 (ADR-0012):  24.8%  +công thức RLC/Coulomb + round-aware scoring
Day-21 (ADR-0017):  16.3%  ⚠️ RESET — chuyển sang dataset chính thức 2026-05-15 (baseline trung thực)
Day-22 (ADR-0023):  27.6%  F1+F2: 6 công thức + routing field-vs-force
Day-25 (ADR-0029):  46.0%  vòng lặp iter-1..7 (sau khi vá 99% leakage)
Day-28 (ADR-0040):  60.7%  iter-18 (6-fix audit batch)
Day-29 (ADR-0041):  66.9%  iter-19 (7-fix tail) ← HEAD hiện tại
```

### Các giai đoạn chính
| ADRs | Giai đoạn | Thành tựu |
| --- | --- | --- |
| 0001–0009 | Nền tảng | Kiến trúc solver-first; logic baseline 35.9%; Z3 backend; unit-aware scoring |
| 0010–0016 | Audit công thức + đóng gói | Resultant forces, RLC, Coulomb vectors; round-aware; Dockerfile + 1-page PDF |
| 0017 | **Migrate dataset chính thức** | 2026-05-15; phát hiện & sửa lỗi data; baseline trung thực 16.3%/22.2% |
| 0018–0028 | RAG, self-consistency, hybrid | NL→FOL rejection loop; N=3 voting; phát hiện **99% leakage** → xây holdout 163-row |
| 0029–0041 | **Vòng lặp iter-driven** | 27.6% → 66.9% physics qua 12+ iteration, mỗi iter ≥3 rows + test + ADR |

### Bài học từ các ADR "âm" (đừng lặp lại)
- **ADR-0013**: SFT-7B chỉ +1.5pp → hệ thống bị giới hạn bởi **solver**, không phải LLM. Đừng đầu tư quá vào SFT.
- **ADR-0022**: SFT-7B hybrid **−6.6pp** trên slice LD dù +2.4pp tổng → không deploy A/B SFT-7B cho domain LD.
- **ADR-0034**: "Unit-Sanity Guard" net-regress cả 5 biến thể (−1.2pp) → đừng làm intent detection kiểu substring.
- **ADR-0036**: MC per-option Z3 đổi 0 prediction nhưng +3× latency → bỏ.

---

## 4. Kiến trúc thực tế (đã verify trong code)

```
POST /predict
  → api/routes.py  →  agent/orchestrator.py  (Orchestrator.predict)
                          │  router.route_task: có "premises-NL" → logic, ngược lại → physics
                          │  (task_type tường minh luôn override)
            ┌─────────────┴─────────────┐
            ▼ logic                      ▼ physics
   logic/pipeline.py               physics/pipeline.py → physics/solver.py
   ├ premise_selector (TF-IDF)     ├ question_cleaner
   ├ rule_parser                   ├ topic_classifier (keyword + symbol heuristic)
   ├ forward_chainer (Jaccard)     ├ quantity_extractor (regex)
   ├ llm_translator (NL→FOL)       ├ llm_extractor (LLM fallback + self-consistency N=3)
   ├ z3_verifier (fallback)        ├ unit_converter (pint → SI)
   ├ answer_verifier               ├ formula_library (70 công thức YAML → SymPy)
   └ explanation                   ├ verifier (sanity magnitude)
                                    ├ rag_retriever (TF-IDF few-shot, E8)
                                    └ explanation
                          ▼
              agent/output_formatter.py → schemas.PredictResponse
                          │
              agent/self_correction.py  (confidence-scored, ≤2 rounds, mặc định OFF local / ON trong Docker)
```

**Điểm cần biết:**
- LLM client là `llm/vllm_client.py` (OpenAI-compatible). Deploy qua **Ollama** (`qwen2.5:3b-instruct`).
- Self-correction **OFF mặc định** local (để xác định) nhưng **ON** trong docker-compose (`EXACT_SELF_CORRECTION__ENABLED=true`).
- Có **hybrid LD client** thứ hai (`EXACT_LLM_LD__*`) cho định tuyến theo domain (ADR-0025/0027).

---

## 5. Phân tích thất bại (cơ hội cải tiến)

### Physics — failure clusters (HANDOVER §7.2, trên 163-row)
| Loại lỗi | Số rows | Hành động đề xuất |
| --- | ---: | --- |
| `no_formula_matched` | **33** | Audit + thêm công thức theo pattern iter-3..19 đã chứng minh |
| `llm_recovery_failed` | **20** | Mạnh hóa `physics/llm_extractor.py` (N-self-consistency, prompt tốt hơn) |
| `missing_input` | **17** | Mở rộng regex role-aware trong `physics/quantity_extractor.py` |

- **NL-prefix là tệ nhất** (Full✓ = 9.5%) — bài prose có setup ngầm, cần một pass extractor riêng.
- _(ANALYSIS bản trước có ghi "type_mismatch 10 rows" và vài % theo loại câu — những số đó **không có nguồn**, đã xóa.)_

### Logic — translator-bound
- ~24% Correct, ~40% abstain. Nút thắt: **3B dịch NL→FOL kém**.
- `fol_parser.py` coverage 48%→51% (ADR-0014). Còn dư địa lớn.

---

## 6. Cơ hội cải tiến (xếp theo ROI)

> ⏱️ **Cửa sổ sửa code thực tế gần như đã đóng.** Giai đoạn thi đấu chính kết thúc **30/5** (hôm qua); cửa sổ nộp lại API **duy nhất** còn lại là **giai đoạn tinh chỉnh 3–4/6** (sau feedback Phase 1 — "cơ hội cuối cùng để cải thiện"). Mọi cải tiến dưới đây chỉ kịp nếu lọt vào 2 ngày đó; sau 4/6 chỉ còn finals trực tiếp 15/6. Lịch đầy đủ + luật + đánh giá đã xác minh qua web: [docs/notes/competition_brief_verified.md](docs/notes/competition_brief_verified.md).

### 6.1 Logic translator — ĐÒN BẨY CAO NHẤT 🔴
24% Correct / 40% abstain là **translator-bound**. Lựa chọn:
- **Few-shot RAG** premise→FOL từ `data/official_v20260515/train/logic_train.jsonl` (đã có `rag_retriever.py` cho physics — tái dùng pattern).
- **LoRA nhỏ** chỉ nhắm NL→FOL (không sinh đáp án).
- Mở rộng vocab `logic/fol_parser.py` (ADR-0014/0017).
- **Mục tiêu:** 24% → 35% Correct (lift tương đương physics 60→70%).

### 6.2 Physics residual 🟠
- 33 rows `no_formula_matched` → thêm 6–8 công thức (mỗi cái ~+1–2%). Ước tính → ~70–75%.
- 20 rows `llm_recovery_failed` → extractor mạnh hơn → +2–3%.

### 6.3 ĐỒNG BỘ TÀI LIỆU (BẮT BUỘC trước khi nộp) 🟡
Repo có **nhiều chỗ tài liệu lệch với code thật** — jury sẽ đọc tài liệu, cần sửa:
| Chỗ lệch | Tài liệu nói | Code thật |
| --- | --- | --- |
| Số công thức | README / architecture.md / **solution_description.md**: **47** | thực tế **70** |
| Số test | README badge: **271** | thực tế **295** hàm `def test_` (pytest gom ~**362** ca do parametrize) |
| Logic score | README/HANDOVER: **~24%** | mới nhất ổn định **28.4%** (ADR-0036) |
| Kiến trúc | `docs/notes/idea.md`: Qwen3-8B + **Qwen-Scope SAE + GRPO** | thực tế Qwen2.5-3B + solver xác định, **không** GRPO/SAE |
| Physics score | README/HANDOVER: ~63% | ADR-0041: 66.9% (cần chạy eval chốt) |
| Eval reports | HANDOVER trỏ `outputs/eval/final_*/...md` | thư mục `outputs/` **không tồn tại** trong repo |
| Docker comment | `docker-compose.yml`: "28.6% physics / 37.5% logic" | đó là điểm Day-8/14; hiện 66.9% / ~28% |
| Default model | `.env.example`: `Qwen/Qwen3-8B` @ vLLM:8001 | deploy override → Ollama `qwen2.5:3b-instruct` |
| Makefile | `make train-grpo` → `exact_agent.train.run_grpo` | module `run_grpo.py` **không tồn tại** (chỉ có prepare_sft.py, run_sft_qwen.py) |

---

## 7. Việc nên làm NGAY (để có nền vững khi vibe-code)

0. **Dựng môi trường — máy hiện CHƯA sẵn sàng** (kiểm tra 2026-05-31: thiếu `uv`, `ollama`, `docker`; Python hệ thống là **3.14** quá mới, nhiều dep khoa học có thể chưa có wheel). Dùng **Python 3.11** (khớp CI + Docker):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
   uv python install 3.11
   uv sync --all-extras && pre-commit install
   # LLM path (cho eval --with-llm / API): brew install ollama && ollama serve & ollama pull qwen2.5:3b
   ```
1. **Chạy eval để có số tươi** (xác nhận 63% vs 66.9%):
   ```bash
   ollama serve & ollama pull qwen2.5:3b
   uv run python scripts/run_eval.py --task physics --with-llm \
     --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
     --out outputs/eval/repro_physics
   uv run python scripts/run_eval.py --task logic --with-llm \
     --split data/official_v20260515/eval_split/logic_eval.jsonl \
     --out outputs/eval/repro_logic
   ```
2. **Xác nhận test xanh:** `uv run pytest -q` (kỳ vọng 295 pass).
3. **Chốt headline number** rồi cập nhật README/HANDOVER/architecture.md (47→70 formulas, 271→295 tests, score).

---

## 8. Những gì CỐ TÌNH không deploy
| Thứ | Lý do | Vị trí |
| --- | --- | --- |
| QLoRA SFT-7B | Single 24.5% (regression), hybrid 30.1% < deterministic 46% | `train/run_sft_qwen.py`, ADR-0013/0022 |
| GRPO (TRL) | Chưa làm — `make train-grpo` trỏ module chưa tồn tại | (stub) |
| Qwen-Scope SAE | Chỉ trong idea.md, không implement | docs/notes/idea.md |
| Qwen3-8B backbone | Day-23 A/B không lift, chọn 3B vì latency | ADR-0022 |

Nếu mở lại bất kỳ thứ nào → đo trên **163-row SFT-unseen holdout** để so sánh công bằng.

---

## 9. Quy trình iteration (đã chứng minh — giữ nguyên)
1. Chạy eval → cluster các row sai theo pattern.
2. Chọn cluster ≥3 rows cho 1 batch iter.
3. Sửa: công thức YAML / routing guard / regex extractor / prompt.
4. Thêm **test riêng cho từng row** đã sửa trong `tests/`.
5. Chạy lại eval, xác nhận lift, viết ADR `docs/decisions/00NN-dayXX-iterYY-*.md`.
6. Commit 3–4 đơn vị semantic: `feat(iter-NN-fmt)`, `feat(iter-NN-route)`, `fix(iter-NN-text)`, `test(iter-NN)`.

**Quy tắc vàng:** Không ship thay đổi làm dịch chuyển con số đo được nếu **không có ADR**.

---

## 10. Tài liệu liên quan
| Đọc khi… | File |
| --- | --- |
| Vibe-code với Claude (file này phục vụ Claude) | [CLAUDE.md](CLAUDE.md) |
| Mới vào dự án | [HANDOVER.md](HANDOVER.md) |
| Cần kiến trúc module-level | [docs/architecture.md](docs/architecture.md) |
| Hiểu vì sao một quyết định | [docs/decisions/](docs/decisions/) (41 ADRs) |
| 1-page submission | [docs/solution_description.md](docs/solution_description.md) |
| Live demo finals | [docs/defense_demo_script.md](docs/defense_demo_script.md) |

---

_Dựng từ việc đọc trực tiếp source ngày 2026-05-31. Mọi con số có nguồn; chỗ lỗi thời đã đánh dấu._
