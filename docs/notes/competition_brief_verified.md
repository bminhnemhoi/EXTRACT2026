# EXACT 2026 — Hồ sơ cuộc thi (đã xác minh qua web, 2026-05-31)

> Tài liệu này hợp nhất bản brief gốc trong repo ([debai.md](debai.md)) với một lượt **deep-research đa nguồn + kiểm chứng đối kháng** (20/25 claim được xác nhận, 5 bị bác bỏ). Mọi mục đều có nguồn ở cuối. Khác với debai.md (viết ở thì tương lai "sẽ…"), bản này diễn giải **ta đang đứng ở đâu trong dòng thời gian ngay lúc này**.

**Tên đầy đủ:** EXACT 2026 — *The 2nd International XAI Challenge for Transparent Educational Question-Answering*
**Sự kiện:** Cuộc thi chính thức của **IEEE IJCNN 2026** (trong khuôn khổ IEEE WCCI 2026, Maastricht).
**Trang chính thức:** https://ura.hcmut.edu.vn/exact · **Liên hệ:** ura.hcmut@gmail.com

---

## ⏱️ Bạn đang ở đâu trong dòng thời gian (tính đến 31/5/2026) — QUAN TRỌNG NHẤT

| | |
| --- | --- |
| ⛔ **Giai đoạn thi đấu chính ĐÃ ĐÓNG** | kết thúc **30/5/2026** (hôm qua). API endpoint nộp tính đến 30/5 là bản được chấm Phase 1. |
| 📊 **Phase 1 — kết quả chấm tự động** | **1–2/6/2026** (sắp tới): chấm độ chính xác + chất lượng giải thích, có feedback chi tiết. |
| 🔧 **Giai đoạn tinh chỉnh mô hình** | **3–4/6/2026** — **CỬA SỔ SỬA CODE / NỘP LẠI API DUY NHẤT CÒN LẠI.** Brief gọi đây là *"cơ hội cuối cùng để cải thiện mô hình trước vòng đánh giá thứ hai."* |
| 📊 **Phase 2 — kết quả chấm** | **5–7/6/2026**: chấm lại P1 + P2 + P3; gộp điểm 2 vòng → bảng xếp hạng sơ bộ. |
| 🏆 **Công bố bảng xếp hạng + Top 10** | **10/6/2026** — Top 10 được mời dự Public Test Day. |
| 🎤 **Finals (Public Test Day)** | **15/6/2026** — chạy hệ thống trực tiếp trên truy vấn chưa từng tiết lộ; ban giám khảo chấm trực tiếp; công bố kết quả cuối. |
| 📄 **Nộp bài nghiên cứu (Top 10)** | **30/6 – 15/7/2026**. |
| 🗣️ **Thuyết trình tại CSoNet 2026** | **16–18/11/2026** (Việt Nam). |

> ⚠️ **Hệ quả cho việc lập kế hoạch:** runway để **thay đổi code** thực tế **không phải "~2 tuần"** — mà gần như đã đóng, chỉ còn **2 ngày (3–4/6)** để nộp lại endpoint sau khi có feedback Phase 1. Mọi cải tiến (logic translator, thêm công thức physics) chỉ kịp áp dụng nếu lọt vào cửa sổ đó. Sau 4/6 chỉ còn finals trực tiếp (15/6).

> 📝 Ghi chú: con số **"còn 32 ngày"** xuất hiện trong [debai.md](debai.md) (dòng 19) và các bản ANALYSIS/HANDOVER cũ là **text snapshot lỗi thời** scrape từ giữa tháng 5 — không phản ánh thực tế ngày 31/5.

---

## 📜 Luật chơi (đã xác minh)

**LÀM:**
- ✅ **Mỗi câu trả lời bắt buộc kèm giải thích ngôn ngữ tự nhiên**, nêu rõ cách suy ra — ngắn gọn, dễ hiểu, kiểm chứng được. *(`answer` + `explanation` là 2 trường BẮT BUỘC.)*
- ✅ **Khuyến khích** dùng công cụ suy luận biểu tượng (Z3, solver tự xây) — *không bắt buộc*. Mọi phương pháp tạo kết quả giải thích được đều hợp lệ.
- ✅ **Chỉ dùng LLM mã nguồn mở ≤ 8 tỷ tham số** — áp dụng cho mọi thành phần LLM (sinh đáp án, suy luận, hay dịch NL→logic).

**KHÔNG ĐƯỢC:**
- ⛔ **Dùng LLM thương mại / mã nguồn đóng** (GPT, Claude, Gemini…) → **bị loại tuyệt đối.**
- ⛔ **Giấu dataset bên ngoài**: mọi dataset dùng để fine-tune LLM/Symbolic Engine **phải công khai đầy đủ** → không công khai = **bị loại.**

**Ai được tham gia:** mở toàn cầu — học sinh THPT, sinh viên, người đi làm, nhà nghiên cứu; không giới hạn tuổi/quốc tịch/đơn vị. **Ngoại lệ:** thành viên Nhóm Nghiên cứu URA (ban tổ chức) không đủ điều kiện.

---

## 🗂️ Dữ liệu

| Loại | Quy mô | Mô tả | Input lúc chấm |
| --- | --- | --- | --- |
| **Type 1 — Logic** | **464 records / 913 câu hỏi** | Suy luận logic trên quy định ĐH (chấm điểm, đăng ký môn, học bổng, yêu cầu học thuật). Dạng: MC, Yes/No/Unknown, open. Mỗi record có premises ở cả NL lẫn FOL + đáp án + giải thích người viết. | câu hỏi **+ `premises-NL`** |
| **Type 2 — Physics** | **5,520 bài toán** | Bài toán số dạng văn bản: mạch điện + tĩnh điện (điện trở, điện áp, dòng, công suất, điện dung, điện trường, năng lượng). Mỗi bài có CoT từng bước + đáp số kèm đơn vị. | **chỉ** câu hỏi |

Bộ test chính thức **gộp cả hai loại** thành một dataset thống nhất. Phân bổ chủ đề + trọng số P1/P2/P3 công bố tại hội thảo khởi động.

> 🔎 *Lưu ý:* các trường FOL/CoT/explanation trong dữ liệu huấn luyện chỉ là **annotation tham chiếu** để làm mẫu — không phải input lúc chấm.

---

## 📊 Tiêu chí đánh giá

| | Tiêu chí | Mô tả |
| --- | --- | --- |
| **P1** | Tính chính xác | Đáp án chính xác, cụ thể (chấm tự động so với ground-truth). |
| **P2** | Chất lượng giải thích | Giải thích NL rõ ràng, mạch lạc, chứng minh được đáp án (ban tổ chức review). |
| **P3** | Chiều sâu lập luận | Bằng chứng hỗ trợ: FOL, các bước CoT, tiền đề, hoặc proof có cấu trúc. Lập luận mạnh hơn → xếp hạng cao hơn (đặc biệt ở finals). |

**Điểm cuối = tổ hợp có trọng số P1 + P2 + P3** (trọng số cụ thể công bố cùng dataset chính thức).
**Quy trình:** Phase 1&2 chấm tự động + review → Finals: chạy trực tiếp trên truy vấn unseen, ban giám khảo chấm thời gian thực.

---

## 📤 Yêu cầu nộp bài

Mỗi đội nộp: **(1) một API endpoint** + **(2) mô tả giải pháp 1 trang** (phương pháp, model, dataset huấn luyện).
Mỗi truy vấn, API trả về:

```jsonc
{
  // BẮT BUỘC
  "answer": "B",
  "explanation": "The voltage across R2 is calculated using ...",
  // TÙY CHỌN (khuyến khích — tăng điểm P3)
  "fol": "∀x (Resistor(x) → HasVoltage(x, V))",
  "cot": ["Step 1: ...", "Step 2: ..."],
  "premises": ["Ohm's law: V = IR", "KVL: sum of voltages in a loop = 0"],
  "confidence": 0.92
}
```

Được nộp nhiều lần; **chỉ bản mới nhất được chấm.** *(Khớp 100% với schema `PredictResponse` của repo này.)*

---

## 🏆 Giải thưởng

- **Top 5:** giải thưởng tiền mặt + mời thuyết trình tại CSoNet 2026 (Việt Nam).
- **Top 10:** mời nộp bài cho Special Session *"Explainable AI for Educational QA"* tại **CSoNet 2026** (Hội nghị quốc tế lần thứ 15 về Computational Social Networks).
- **Mọi đội có bài hợp lệ qua Phase 1&2:** giấy chứng nhận tham gia chính thức.

---

## 👥 Ban tổ chức & Hội đồng

**Tổ chức bởi:** Nhóm Nghiên cứu **URA**, Đại học Bách khoa TP.HCM (**HCMUT**), phối hợp Đại học Naples Parthenope (Ý).
**Senior organizers:** GS. Angelo Ciaramella (Naples Parthenope) · Ô. Nguyễn Song Thiên Long (HCMUT).
**Competition Chairs (hội đồng quốc tế):** GS. Quân Thành Thọ (HCMUT) · GS. Emanuel Di Nardo (Naples Parthenope, Ý) · GS. Nguyễn Đức Anh (USN, Na Uy) · GS. Fabien Baldacci (Bordeaux, Pháp) · GS. Nguyễn Lê Minh (JAIST, Nhật).

**Quy mô tham gia (snapshot brief):** ~**180 đội / 438 người / 6 quốc gia** (chủ yếu Việt Nam: 174). Top đơn vị: HCMUT (97), HCMUS (70), UIT (42).

---

## 📚 Phiên bản trước (IJCNN 2025)

Phiên bản 1 tổ chức tại workshop **TRNS-AI** (Trustworthiness & Reliability in Neuro-symbolic AI), cùng IEEE IJCNN 2025 — **107 người / 30 đội**, tạo benchmark công khai đầu tiên cho explainable academic QA.
EXACT 2026 mở rộng phạm vi từ quy định giáo dục **sang lập luận STEM** (physics) và thêm tiêu chí đánh giá có cấu trúc.
Paper 2025: *"Bridging LLMs and Symbolic Reasoning in Educational QA Systems: Insights from the XAI Challenge at IJCNN 2025"*, ITADATA 2025 — arXiv [2508.01263](https://arxiv.org/abs/2508.01263). Trang 2025: https://sites.google.com/view/trns-ai/challenge

> ⚠️ **Đừng nhầm số liệu 2025 với 2026:** web research có gặp claim "dataset 481 train / 50 test" và một số metric Exact-Match của bản 2025 — **những con số đó thuộc edition 2025, KHÔNG áp dụng cho 2026** (2026: logic 464/913, physics 5,520). Các claim này đã bị lượt kiểm chứng bác bỏ khi gán cho 2026.

---

## ✅ Web-research đã xác minh / bác bỏ điều gì

**Xác nhận cao (3-0 hoặc 2-1) — khớp với debai.md:**
- EXACT 2026 = lần thứ 2, thuộc IEEE IJCNN 2026. ✓
- Dòng thời gian: đăng ký 10/4–10/5; thi đấu chính 5–30/5; finals 15/6. ✓
- Luật ≤8B open-source, cấm GPT/Claude/Gemini (bị loại). ✓
- `answer` + `explanation` bắt buộc; `fol`/`cot`/`premises`/`confidence` tùy chọn. ✓
- Đánh giá P1/P2/P3 có trọng số. ✓
- Mở toàn cầu, trừ thành viên URA. ✓

**Bị bác bỏ (không áp dụng cho 2026):**
- ✗ "Dataset 481 train / 50 test" → là edition 2025.
- ✗ Các metric Exact-Match / "panel giáo sư chấm P3" mô tả protocol 2025, không phải 2026.

**Câu hỏi còn mở (web research không tìm thấy):**
- Kết quả cuối / đội thắng / kiến trúc thắng cuộc EXACT 2026 — chưa công bố (cuộc thi vừa mới kết thúc giai đoạn chính).
- Trọng số chính xác P1/P2/P3 — công bố cùng dataset chính thức (không có công khai).

---

## 🔗 Nguồn

- **Chính thức:** https://ura.hcmut.edu.vn/exact (trang cuộc thi EXACT 2026)
- **IEEE WCCI/IJCNN 2026:** https://attend.ieee.org/wcci-2026/ (registrations, important-dates, topics)
- **Brief gốc trong repo:** [debai.md](debai.md) (scrape trang chính thức, bản tiếng Việt — chi tiết nhất)
- **Edition 2025:** arXiv [2508.01263](https://arxiv.org/abs/2508.01263) · https://sites.google.com/view/trns-ai/challenge

_Tổng hợp 2026-05-31 từ deep-research (5 góc tìm kiếm, 15 nguồn, 52 claim → 25 kiểm chứng → 20 xác nhận) + đối chiếu debai.md._
