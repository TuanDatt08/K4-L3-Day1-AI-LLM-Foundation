# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Theo mình thì ở temperature 0.0, phản hồi thường ổn định, và có cách diễn đạt gần như giống nhau giữa các lần chạy. Khi tăng lên 0.5, 1.0 và 1.5, thì các cách dùng từ, ví dụ và cấu trúc câu sẽ đa dạng và sáng tạo hơn; temperature càng cao thì phản hồi càng sáng tạo nhưng cũng có thể lan man và kém ổn định hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ chọn temperature khoảng 0.2–0.4 cho chatbot hỗ trợ khách hàng. Mức này giúp câu trả lời nhất quán, chính xác và ít đưa ra thông tin linh tinh, trong khi vẫn đủ tự nhiên để giao tiếp với khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Vì cùng có 350 token đầu ra mỗi lần gọi, GPT-4o có giá output 0.010 USD/1K token còn GPT-4o-mini là 0.0006 USD/1K token, nên GPT-4o đắt khoảng 16,7 lần. Với 10.000 người dùng và 3 lượt/ngày, tổng là 105 triệu token output/ngày; GPT-4o phù hợp cho phân tích phức tạp hoặc yêu cầu độ chính xác cao, còn mini phù hợp cho phân loại yêu cầu và trả lời thường ngày để giảm chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với system prompt dành cho giáo viên tiểu học, câu trả lời thường ngắn hơn, dùng từ vựng đơn giản, phép so sánh gần gũi và ví dụ dễ hình dung. Với system prompt chuyên gia tài chính, câu trả lời thường dài hơn, dùng các thuật ngữ chuyên ngành, từ vựng chuyên sâu hơn. System prompt định hướng vai trò, đối tượng, giọng điệu và mức độ chuyên sâu của model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với một đoạn tiếng Việt khoảng 100 từ, ví dụ tiktoken đếm được 145 token, còn công thức số từ / 0.75 ước tính khoảng 133 token; chênh lệch là khoảng 9% so với ước tính. Hai con số khác nhau vì tokenizer không nhất thiết coi mỗi từ là một token: dấu câu, khoảng trắng, từ có dấu và các phần của từ có thể được tách riêng. Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì cách phân tách từ và dữ liệu huấn luyện của tokenizer phù hợp với tiếng Anh hơn, đặc biệt với ký tự có dấu.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng khi người dùng phải chờ câu trả lời dài, chẳng hạn chatbot, trợ lý học tập hoặc công cụ viết nội dung.Non-streaming phù hợp khi cần toàn bộ kết quả trước khi xử lý bước tiếp theo, khi lưu một response hoàn chỉnh vào cơ sở dữ liệu, hoặc khi phản hồi rất ngắn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff làm các lần retry sau cách xa nhau hơn, nhờ đó giảm áp lực lên API đang quá tải và cho hệ thống thời gian phục hồi. Nếu hàng nghìn client cùng retry sau đúng một khoảng cố định, chúng sẽ tạo ra các đợt request đồng loạt, tiếp tục làm nghẽn API và dễ gây hiệu ứng “thundering herd”. Có thể kết hợp thêm jitter ngẫu nhiên để các client không retry cùng thời điểm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona mình chọn là trợ giảng lập trình thân thiện: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.” Cụm “bằng tiếng Việt” giúp đầu ra phù hợp với người học.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ ba lượt gần nhất, nên trợ lý có thể quên mục tiêu hoặc quyết định ở những lượt cũ. Một cải thiện cụ thể là thêm bộ nhớ tóm tắt: sau mỗi vài lượt, dùng model tạo một bản tóm tắt ngắn về mục tiêu, sở thích và các quyết định quan trọng, sau đó đưa bản tóm tắt này vào system context cùng history hiện tại.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
