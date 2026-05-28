Dưới đây là **bản ý tưởng final** sau khi tổng hợp toàn bộ phiên làm việc: yêu cầu cuộc thi, 2 bộ dữ liệu thật, dữ liệu đã clean, hướng nghiên cứu mới như Qwen-Scope, GRPO, Logic-LM, FoVer, Physics Reasoner, và chiến lược triển khai thực tế.

---

# 1. Tên ý tưởng final

## **VeriScope-Agent**

### **A Solver-Verified Qwen-Scope Agent for Explainable Educational QA**

Tên tiếng Việt:

> **VeriScope-Agent: Hệ thống giáo viên AI giải thích được, kết hợp Qwen3-8B, Qwen-Scope, symbolic solver và agent tự kiểm chứng**

---

# 2. Tư tưởng cốt lõi

Ý tưởng final không phải là “fine-tune LLM để nó tự trả lời”. Hướng mạnh nhất là:

```text
Qwen3-8B hiểu đề, lập kế hoạch, sinh giải thích
+
Logic solver / Physics solver quyết định đáp án
+
Verifier kiểm tra lại kết quả
+
Agent tự sửa nếu sai
+
Qwen-Scope hỗ trợ XAI/steering ở mức representation
```

Nói ngắn gọn:

> **LLM không được quyền đoán đáp án cuối. Solver quyết định đáp án. LLM giải thích lại bằng ngôn ngữ tự nhiên từ trace đã kiểm chứng.**

Đây là điểm rất quan trọng vì dataset gốc có nhiều lỗi nhãn: có câu `answer = Unknown` nhưng `explanation` lại chứng minh option cụ thể là đúng; ví dụ ngay đầu file logic, explanation nói “supporting option A” trong khi answer gốc là `Unknown`. 

---

# 3. Dữ liệu hiện tại đã được clean như thế nào?

Bạn hiện có package clean:

**[Download cleaned_exact_dataset_package.zip](sandbox:/mnt/data/cleaned_exact_dataset_package.zip)**

Các file nên dùng:

| File                                                                                                             | Vai trò                                    |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| [sft_train_mixed_solver_clean.jsonl](sandbox:/mnt/data/cleaned_exact_dataset/sft_train_mixed_solver_clean.jsonl) | Dataset mixed Logic + Physics để fine-tune |
| [logic_train_safe.jsonl](sandbox:/mnt/data/cleaned_exact_dataset/logic_train_safe.jsonl)                         | Logic samples an toàn hơn                  |
| [physics_train_safe.jsonl](sandbox:/mnt/data/cleaned_exact_dataset/physics_train_safe.jsonl)                     | Physics samples an toàn hơn                |
| [physics_train_safe.csv](sandbox:/mnt/data/cleaned_exact_dataset/physics_train_safe.csv)                         | Physics bản CSV                            |
| [cleaning_report.json](sandbox:/mnt/data/cleaned_exact_dataset/cleaning_report.json)                             | Báo cáo cleaning                           |

Bản clean hiện tại nên được xem là:

```text
cleaned training subset
+ quality flags
+ suspect samples tách riêng
+ SFT-ready mixed dataset
```

Điểm quan trọng: **data cleaning không phải đóng góp chính của dự án**, nhưng là bước preprocessing bắt buộc để tránh fine-tune vào nhãn sai.

---

# 4. Cơ sở nghiên cứu cho ý tưởng final

## 4.1. Logic-LM: LLM + symbolic solver

Logic-LM đề xuất dùng LLM để dịch bài toán ngôn ngữ tự nhiên sang biểu diễn symbolic, sau đó để solver deterministic suy luận. Paper báo cáo cải thiện trung bình 39.2% so với prompting thường và 18.4% so với CoT prompting trên các benchmark logic. Đây là nền tảng rất sát với phần Logic-Based Educational Queries. ([ACL Anthology][1])

## 4.2. FoVer: Natural language → FOL → Z3 verification

FoVer dùng LLM để chuyển natural language reasoning thành biểu thức FOL executable, rồi dùng Z3 theorem prover để kiểm chứng. Đây là hướng rất hợp cho dataset logic của EXACT vì hệ thống cần chứng minh answer dựa trên premises. ([ACL Anthology][2])

## 4.3. Physics Reasoner: formula set + checklist + guided reasoning

Physics Reasoner chỉ ra LLM dễ sai vì thiếu kiến thức vật lý hoặc áp dụng sai công thức. Framework của họ gồm 3 bước: problem analysis, formula retrieval, guided reasoning; đồng thời dùng formula set và checklist để hướng dẫn áp dụng kiến thức. Đây là nền tảng trực tiếp cho pipeline Physics của mình. ([arXiv][3])

## 4.4. GRPO: tối ưu reasoning bằng reward kiểm chứng được

DeepSeekMath giới thiệu GRPO như một biến thể của PPO giúp cải thiện mathematical reasoning và tối ưu memory hơn PPO. Hugging Face TRL hiện hỗ trợ `GRPOTrainer`, cho phép dùng nhiều reward functions và weighted rewards. ([arXiv][4])

Với cuộc thi này, GRPO hợp vì ta có reward kiểm chứng được:

```text
answer đúng hay sai
unit đúng hay sai
formula đúng hay sai
proof có hợp lệ không
JSON format có đúng không
explanation có bám trace không
```

## 4.5. Qwen-Scope: SAE cho XAI và steering

Qwen-Scope là paper mới ngày 12/05/2026, giới thiệu bộ Sparse Autoencoders open-source cho Qwen3/Qwen3.5. Paper nói Qwen-Scope có 14 nhóm SAE trên 7 model variants và dùng được cho 4 hướng: inference-time steering, evaluation analysis, data-centric workflows, và post-training optimization. ([arXiv][5])

Điểm cần hiểu đúng:

```text
Qwen-Scope không thay solver.
Qwen-Scope không phải “đọc suy nghĩ thật” tuyệt đối.
Qwen-Scope là representation-level diagnostic/steering layer.
```

Tức là nó giúp mình phân tích và điều chỉnh hành vi model, ví dụ giảm repetition, giảm code-switching, tăng structured reasoning, nhưng **answer cuối vẫn phải do solver/verifier quyết định**.

## 4.6. Vì sao chọn Qwen3-8B?

Qwen3-8B phù hợp vì là open-weight, nằm trong giới hạn ≤8B, có năng lực reasoning, instruction-following, agent và multilingual. ([Hugging Face][6]) Ngoài ra, Qwen-Scope có SAE checkpoint trực tiếp cho Qwen3-8B như `SAE-Res-Qwen3-8B-Base-W64K-L0_50`, được mô tả là có thể dùng cho steerable inference, evaluation analysis, data classification/synthesis và model optimization. ([Hugging Face][7])

---

# 5. Kiến trúc hệ thống final

```text
Input JSON
  ↓
Input Normalizer
  ↓
Task Router
  ├── Logic QA Pipeline
  │     ├── Premise Selector
  │     ├── NL-to-Rule Parser
  │     ├── Forward-Chaining Solver
  │     ├── Z3/FOL Verifier
  │     ├── Multiple-Choice Claim Verifier
  │     ├── Yes/No/Unknown Verifier
  │     └── Proof Trace Builder
  │
  └── Physics QA Pipeline
        ├── Question Cleaner
        ├── Topic Classifier
        ├── Quantity Extractor
        ├── Unit Converter
        ├── Formula Retriever
        ├── SymPy/Python Solver
        ├── Numeric/Unit Verifier
        └── Formula Trace Builder
  ↓
Agentic Self-Correction Loop
  ↓
Qwen-Scope SAE Monitor / Optional Steering
  ↓
Faithful Explanation Generator
  ↓
Confidence Estimator
  ↓
JSON Output
```

---

# 6. Pipeline Logic-Based Educational Queries

## Input

```json
{
  "premises-NL": [
    "If a student completes required courses, they are eligible for graduation.",
    "If a student is eligible and has GPA above 3.5, they graduate with honors.",
    "John completed required courses.",
    "John has GPA 3.8."
  ],
  "question": "Does John graduate with honors?"
}
```

## Xử lý

### Bước 1: Premise selection

Chọn premise liên quan:

```text
P1, P2, P3, P4
```

### Bước 2: Parse thành facts/rules

```text
P1: completed_required_courses(x) → eligible_for_graduation(x)
P2: eligible_for_graduation(x) ∧ gpa_above_3_5(x) → graduates_with_honors(x)
P3: completed_required_courses(John)
P4: gpa_above_3_5(John)
```

### Bước 3: Solver suy luận

```text
P3 + P1 → eligible_for_graduation(John)
eligible_for_graduation(John) + P4 + P2 → graduates_with_honors(John)
```

### Bước 4: Sinh explanation từ proof trace

Output:

```json
{
  "answer": "Yes",
  "explanation": "Premise 3 states that John completed the required courses, and premise 1 implies that he is eligible for graduation. Premise 4 states that his GPA is above 3.5, and premise 2 implies that eligible students with GPA above 3.5 graduate with honors. Therefore, John graduates with honors.",
  "cot": [
    "P3 + P1 -> eligible_for_graduation(John)",
    "eligible_for_graduation(John) + P4 + P2 -> graduates_with_honors(John)"
  ],
  "premises": ["P1", "P2", "P3", "P4"],
  "confidence": 0.94
}
```

---

# 7. Pipeline Physics Problems

## Input

```json
{
  "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
}
```

## Xử lý

### Bước 1: Quantity extraction

```text
C = 100 μF
U = 30 V
target = energy
```

### Bước 2: Unit conversion

```text
100 μF = 100 × 10^-6 F = 1e-4 F
```

### Bước 3: Formula retrieval

```text
E = 0.5 C U²
```

### Bước 4: Solver tính toán

```text
E = 0.5 × 1e-4 × 30²
E = 0.045 J
```

### Bước 5: Verifier

```text
unit = J đúng
formula đúng
scale hợp lý
```

Output:

```json
{
  "answer": "0.045 J",
  "explanation": "Given C = 100 μF = 1e-4 F and U = 30 V. The energy stored in a capacitor is E = 0.5CU². Substituting gives E = 0.5 × 1e-4 × 30² = 0.045 J. Therefore, the answer is 0.045 J.",
  "cot": [
    "Extract C = 100 μF and U = 30 V.",
    "Convert C to SI units: 100 μF = 1e-4 F.",
    "Use the capacitor energy formula E = 0.5CU².",
    "Compute E = 0.045 J."
  ],
  "premises": [
    "Capacitor energy formula: E = 0.5CU²",
    "100 μF = 1e-4 F"
  ],
  "confidence": 0.98
}
```

---

# 8. Vai trò của dữ liệu clean trong training

Bây giờ data đã clean, chiến lược training nên là:

```text
Không train trên raw data.
Train trên cleaned safe data.
Dùng suspect data cho stress test hoặc uncertain/no-answer sau khi review.
```

Dùng file:

```text
sft_train_mixed_solver_clean.jsonl
```

cho SFT đầu tiên.

## Training target

Mỗi sample nên có dạng:

```json
{
  "prompt": {
    "task_type": "physics",
    "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
  },
  "completion": {
    "answer": "0.045 J",
    "explanation": "...",
    "cot": [...],
    "premises": [...],
    "confidence": 0.98
  }
}
```

Mục tiêu của SFT không phải bắt model học thuộc đáp án, mà là học:

```text
format output
cách diễn đạt explanation
cách trình bày cot
cách gọi/tuân thủ trace
cách trả JSON ổn định
```

---

# 9. Chiến lược fine-tuning final

## Stage 1 — SFT / Solver-Trace Distillation

Model:

```text
Qwen3-8B QLoRA
hoặc Qwen3-4B nếu Colab yếu hơn
```

Data:

```text
sft_train_mixed_solver_clean.jsonl
```

Mục tiêu:

```text
model học output format + explanation style + reasoning trace style
```

Không yêu cầu model tự tính giỏi ngay, vì solver vẫn kiểm chứng.

---

## Stage 2 — GRPO với verifiable rewards

Chỉ chạy sau khi Stage 1 ổn.

### Reward cho Physics

```text
R = 0.40 answer_correct
  + 0.20 unit_correct
  + 0.15 formula_grounded
  + 0.10 explanation_grounded
  + 0.10 json_format_valid
  + 0.05 qwen_scope_behavior_bonus
```

### Reward cho Logic

```text
R = 0.35 solver_answer_match
  + 0.20 premise_selection_match
  + 0.20 proof_consistency
  + 0.10 no_hallucinated_premise
  + 0.10 json_format_valid
  + 0.05 qwen_scope_behavior_bonus
```

Lưu ý: `qwen_scope_behavior_bonus` chỉ là phụ. Reward chính phải đến từ solver/verifier.

---

## Stage 3 — Agentic self-correction

Ở inference:

```text
1. Agent sinh draft.
2. Solver/verifier kiểm tra.
3. Nếu sai, verifier trả feedback.
4. Agent sửa lại.
5. Nếu pass, sinh final answer + explanation.
```

Ví dụ feedback:

```text
The capacitance conversion is wrong. 100 μF should be 1e-4 F, not 100 F. Recompute E = 0.5CU².
```

Đây chính là cơ chế “thầy hỏi lại để học sinh nhận ra sai” mà bạn brainstorm, nhưng formal hóa bằng verifier.

---

## Stage 4 — Qwen-Scope SAE monitor/steering

Qwen-Scope dùng để:

```text
1. Theo dõi activation khi model sinh answer.
2. Phát hiện hành vi không mong muốn: repetition, code-switching, format drift.
3. Steering nhẹ để tăng structured reasoning.
4. Tạo điểm nhấn XAI ở final.
```

Không nên nói:

```text
Mô hình tiết lộ suy nghĩ thật.
```

Nên nói:

```text
We use Qwen-Scope as a representation-level diagnostic and steering module.
```

---

# 10. Điểm mới của dự án

Dự án có 6 điểm mới/chặt chẽ:

```text
1. Unified API xử lý cả Logic và Physics.
2. Dữ liệu đã clean và tách safe/suspect để tránh học nhãn sai.
3. Dual-solver architecture: Z3/rule engine cho Logic, SymPy/formula solver cho Physics.
4. Solver-generated traces làm nguồn distillation sạch.
5. GRPO với reward kiểm chứng được, không reward mù theo label nhiễu.
6. Qwen-Scope SAE làm lớp XAI/steering ở representation level.
```

---

# 11. Bản mô tả ý tưởng final — tiếng Việt

Bạn có thể dùng nguyên đoạn này cho báo cáo/slide/proposal:

```text
Chúng tôi đề xuất VeriScope-Agent, một hệ thống hỏi đáp giáo dục giải thích được cho EXACT 2026, xử lý đồng thời hai dạng dữ liệu Logic-Based Educational Queries và Physics Problems thông qua một API duy nhất. Thay vì để LLM tự sinh đáp án trực tiếp, hệ thống sử dụng Qwen3-8B làm agent lõi để hiểu câu hỏi, lập kế hoạch và sinh giải thích; còn đáp án cuối được quyết định bởi các bộ solver có thể kiểm chứng.

Đối với nhóm câu hỏi logic, hệ thống chọn các premise liên quan, chuyển chúng thành facts/rules, suy luận bằng forward-chaining và dùng Z3/FOL verifier cho các trường hợp có phủ định, lượng từ hoặc ràng buộc phức tạp. Với câu multiple-choice, từng lựa chọn được chuyển thành một claim và kiểm chứng độc lập. Với câu Yes/No/Unknown, hệ thống xác định answer dựa trên entailment, contradiction hoặc insufficient evidence.

Đối với nhóm bài vật lý, hệ thống làm sạch câu hỏi, phân loại chủ đề, trích xuất đại lượng, chuẩn hóa đơn vị, truy hồi công thức từ thư viện công thức vật lý và giải bằng SymPy/Python. Bộ verifier kiểm tra lại kết quả số, đơn vị, công thức, dữ kiện thiếu và các trường hợp mơ hồ. Explanation cuối cùng được sinh từ formula trace hoặc proof trace đã được kiểm chứng, không phải hallucinated chain-of-thought.

Do dataset gốc có nhiều lỗi annotation như mâu thuẫn giữa answer và explanation, FOL sai cú pháp, thiếu answer/unit và một số lỗi tính toán vật lý, chúng tôi sử dụng bản cleaned dataset đã tách safe/suspect samples. Fine-tuning không được thực hiện trên nhãn thô mà dựa trên solver-verified traces để huấn luyện model trả lời đúng format, giải thích rõ ràng và bám sát reasoning trace.

Sau giai đoạn SFT, hệ thống có thể được tối ưu bằng GRPO với verifiable rewards gồm answer correctness, unit correctness, proof consistency, explanation faithfulness và JSON format validity. Ngoài ra, chúng tôi tích hợp Qwen-Scope SAE như một lớp XAI cấp representation để phân tích và điều chỉnh hành vi model như structured reasoning, code-switching hoặc repetition. API cuối cùng trả về answer, explanation, cot, premises, optional FOL/formula trace và confidence, tối ưu đồng thời cho Correctness, Explanation Quality và Reasoning Depth.
```

---

# 12. Bản mô tả ý tưởng final — tiếng Anh

```text
We propose VeriScope-Agent, a solver-verified Qwen-Scope agent for explainable educational question answering in EXACT 2026. The system is designed to handle both logic-based educational queries and physics problems through a single unified API endpoint. Instead of relying on a black-box LLM to directly generate answers, VeriScope-Agent uses Qwen3-8B as a lightweight open-source reasoning and explanation agent, while final answers are determined and verified by symbolic and numerical solvers.

For logic-based queries, the system selects relevant natural-language premises, converts them into normalized facts and rules, performs forward-chaining inference, and applies Z3/FOL verification when negation, quantifiers, or constraints are involved. Multiple-choice options are verified independently as logical claims, and Yes/No/Unknown questions are answered through entailment, contradiction, or insufficient evidence.

For physics problems, the system cleans noisy question text, classifies the physics topic, extracts quantities, normalizes units, retrieves formulas from a curated physics formula library, and solves equations using SymPy/Python. A verifier checks numerical correctness, unit consistency, missing variables, ambiguous geometry, and multi-target prompts. Final explanations are generated from verified proof traces or formula traces to ensure faithfulness.

Because the released datasets contain noisy labels, answer-explanation contradictions, malformed FOL strings, missing answer/unit fields, and incorrect physics annotations, we use a cleaned safe training subset and separate suspicious samples for review. Fine-tuning is performed on solver-verified traces rather than raw labels, enabling the model to learn robust JSON formatting, explanation style, and trace-grounded reasoning.

After supervised fine-tuning, the system can be further optimized using GRPO with verifiable rewards, including answer correctness, unit correctness, proof consistency, explanation faithfulness, and JSON format validity. We additionally integrate Qwen-Scope sparse autoencoder diagnostics as a representation-level XAI and steering module to analyze and reduce undesirable behaviors such as repetition, code-switching, and format drift. The final API returns answer, explanation, reasoning steps, supporting premises, optional FOL/formula traces, and confidence scores, targeting correctness, explanation quality, and reasoning depth.
```

---

# 13. Cách trình bày trong final round

Khi demo với ban giám khảo, bạn nên cho họ thấy 5 lớp:

```text
1. Input question
2. Task router nhận diện Logic hay Physics
3. Solver trace:
   - Logic: selected premises + proof chain
   - Physics: variables + unit conversion + formula + calculation
4. Verifier result:
   - pass/fail
   - nếu fail thì self-correction
5. Final JSON output
```

Và nếu có Qwen-Scope:

```text
6. Qwen-Scope diagnostic:
   - representation-level signal
   - format/repetition/code-switching monitoring
   - optional steering
```

Đây là điểm giúp dự án khác các team chỉ fine-tune hoặc prompt LLM.

---

# 14. Lộ trình triển khai ngay sau khi đã có data clean

## Việc 1 — SFT baseline

Dùng:

```text
sft_train_mixed_solver_clean.jsonl
```

Train Qwen3-4B hoặc Qwen3-8B bằng QLoRA.

Mục tiêu:

```text
JSON output ổn định
explanation tốt
cot/premises đúng format
```

## Việc 2 — Xây solver inference

Không chờ model xong mới làm. Làm song song:

```text
logic_solver/
physics_solver/
verifier/
```

## Việc 3 — Kết hợp agent self-correction

Sau khi có model và solver:

```text
draft → verify → revise → final
```

## Việc 4 — GRPO nhỏ

Chỉ chạy trên clean subset.

Mục tiêu:

```text
tăng answer consistency
tăng explanation faithfulness
giảm format lỗi
```

## Việc 5 — Qwen-Scope demo/steering

Tích hợp sau cùng để tăng P3 và độ ấn tượng final.

---

# 15. Chốt ý tưởng final trong 30 giây

```text
VeriScope-Agent là một giáo viên AI giải thích được dùng Qwen3-8B và Qwen-Scope, nhưng không để LLM tự đoán đáp án. Với Logic, hệ thống chọn premise, chuyển thành rule/FOL và suy luận bằng forward chaining/Z3. Với Physics, hệ thống trích xuất biến, đổi đơn vị, chọn công thức và giải bằng SymPy. Dữ liệu gốc đã được clean để tránh học nhãn sai; model được SFT trên solver-verified traces. Sau đó dùng GRPO với reward kiểm chứng được để tăng correctness và explanation faithfulness. Qwen-Scope được dùng như lớp XAI/steering để phân tích và ổn định reasoning behavior. Output cuối gồm answer, explanation, cot, premises, optional FOL/formula trace và confidence.
```

Đây là bản ý tưởng final chặt nhất: **khả thi, đúng luật, tận dụng data đã clean, có solver để thắng P1, có explanation trace để thắng P2, và có Qwen-Scope/GRPO để tạo lợi thế P3 ở vòng final.**

[1]: https://aclanthology.org/2023.findings-emnlp.248/?utm_source=chatgpt.com "Logic-LM: Empowering Large Language Models with ..."
[2]: https://aclanthology.org/2025.tacl-1.61/?utm_source=chatgpt.com "FoVer: First-Order Logic Verification for Natural Language ..."
[3]: https://arxiv.org/abs/2412.13791?utm_source=chatgpt.com "Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems with Large Language Models"
[4]: https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com "DeepSeekMath: Pushing the Limits of Mathematical ..."
[5]: https://arxiv.org/abs/2605.11887?utm_source=chatgpt.com "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models"
[6]: https://huggingface.co/Qwen/Qwen3-8B?utm_source=chatgpt.com "Qwen/Qwen3-8B"
[7]: https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50?utm_source=chatgpt.com "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
