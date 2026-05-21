# Day-26 — SFT-7B retrain trên data 2026-05-15 (cho user)

Mục tiêu: tạo adapter SFT mới train trên data CHÍNH THỨC 2026-05-15
(1750 train + 194 val) thay vì data 2026-05-09 cũ (1765 train + 196
val). Adapter mới có thể không còn LD regression (đã thấy Day-14
Coulomb vector formulas trong train data) → hybrid lift có thể cao
hơn +2.5pp đã đo Day-23.

**Tổng thời gian active của bạn: ~90 phút.** Chia 3 phase: 5 min
chuẩn bị, 60 min train (đợi), 25 min download + share.

## Phase 1 — Chuẩn bị (5 phút)

Trên máy bạn:

```powershell
cd d:\Exact2026\exact-veriscope-agent
# (Đã được tôi chạy sẵn, nhưng nếu cần re-generate)
uv run python scripts/prepare_sft_data_v20260515.py
```

Output 2 file:
- `data/processed/sft_chat_train_v20260515.jsonl` (1750 records)
- `data/processed/sft_chat_val_v20260515.jsonl` (194 records)

Khi upload lên Colab cell 3 **bạn RENAME chúng** thành:
- `sft_chat_train.jsonl`
- `sft_chat_val.jsonl`

(Notebook hard-code những tên đó. Đổi tên trong dialog upload là OK.)

## Phase 2 — Train trên Colab (5 min setup + 60 min đợi)

1. **Browser private/incognito**, đăng nhập Google account khác
   (acc bạn đã dùng Day-23 không bị tốn quota lại; có thể là acc
   gmail thứ 2 hoặc 3).

2. `https://colab.research.google.com`

3. **File → Upload notebook** → chọn
   `d:\Exact2026\exact-veriscope-agent\notebooks\sft_qwen_colab.ipynb`
   (**lưu ý: notebook TRAIN khác notebook SERVE đã dùng hôm trước**)

4. **Runtime → Change runtime type → T4 GPU → Save**

5. **Chạy lần lượt các cell:**

   | Cell | Việc | Đợi | Output mong đợi |
   |---|---|---|---|
   | **Install** | Click ▶ | ~1 min | `Unsloth installed` etc., no errors |
   | **Upload data** | Click ▶ → "Chọn tệp" dialog xuất hiện | ~30s | Chọn **CẢ HAI** file: train_v20260515 (rename thành `sft_chat_train.jsonl`) + val_v20260515 (rename thành `sft_chat_val.jsonl`). Sau upload: in `Loaded 1750 train + 194 val.` |
   | **Load model** | Click ▶ | ~3-5 min | Tải Qwen2.5-7B base (~5GB). Cuối in `Trainable params: ~16M` (LoRA rank 16) |
   | **Train** | Click ▶ | **~30-60 min** | Loss giảm dần từ ~1.5 → ~0.4. **KHÔNG đóng tab.** |
   | **Save adapter** | Click ▶ | ~30s | Tạo zip `exact_qwen25_7b_lora_v20260515.zip` |
   | **Download** | Click ▶ | ~1-2 min | Browser auto-download zip (~151 MB) về máy bạn |

   **Lỗi thường gặp:**
   - `OOM CUDA out of memory`: bạn vẫn còn cells khác giữ VRAM → Runtime → Restart runtime → chạy lại
   - `Tokenizer error`: cell `Load model` thường tự fix (notebook dùng `inspect.signature` để pick đúng kwarg name — đã debug Day-13)
   - Mất tunnel kết nối: refresh tab, run lại từ cell hiện tại
   - Colab disconnect sau ~90 min idle: KHÔNG để máy sleep, click chuột vào tab mỗi 10-15 min

## Phase 3 — Share adapter cho tôi (25 phút)

Sau khi cell `Download` xong, file `exact_qwen25_7b_lora_v20260515.zip`
xuất hiện trong thư mục Downloads của bạn (~151 MB).

**Cách 1 — Google Drive (recommended):**
1. Upload file lên Google Drive (~1-2 min)
2. Right-click file → Share → "Anyone with the link can view" → Copy link
3. Paste link cho tôi

**Cách 2 — Lưu vào D:\Exact2026:**
1. Move file vào `D:\Exact2026\exact_qwen25_7b_lora_v20260515.zip`
2. Báo cho tôi "đã lưu vào D:\"

**Cách 3 — Re-upload lên serve_sft_colab:**
1. Mở `notebooks/serve_sft_colab.ipynb` trên Colab acc khác (acc thứ 3)
2. Upload zip mới vào cell 2 + chạy hết → tunnel URL mới
3. Paste URL cho tôi

(Cách 3 nhanh hơn nếu bạn muốn tôi A/B test ngay; Cách 1 cho tôi lưu
adapter để dùng deploy sau.)

## Day-27 (sau khi tôi có adapter mới)

Tôi sẽ:
1. A/B eval adapter MỚI vs CŨ trên cùng 163-row SFT-unseen holdout
2. Pick adapter tốt hơn → đưa vào kế hoạch deploy hybrid
3. Update ADR documents kết quả thật

## Sự thật về expected lift

Adapter cũ (Day-13, train trên 2026-05-09) cho **+2.5pp** stacking
trên top of F1+F2. Adapter mới train trên 2026-05-15 có thể:

- **Best case**: +4-6pp (không còn LD regression vì train data có
  Day-14 Coulomb vector formulas). Top-5 odds 35-45%.
- **Realistic**: +2-3pp (similar to cũ). Top-5 odds 30-40%.
- **Worst case**: 0 or regress (hyperparam chưa tune lại). Then ta
  giữ adapter cũ.

Đáng làm vì **upside lớn, cost chỉ 90 min bạn time + Colab free.**
