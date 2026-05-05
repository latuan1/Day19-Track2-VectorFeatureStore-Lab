# Reflection - Lab 19

**Tên:** Lương Anh Tuấn
**Cohort:** A20-K1
**Path đã chạy:** Lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên Precision@10, hybrid thắng trung bình: 78.6%, vì RRF gom được tín hiệu từ cả BM25 và vector. Với `exact`, BM25 và hybrid gần như hòa ở 96.7% vì query chứa đúng keyword kỹ thuật trong corpus, nên lexical match đã rất mạnh. Với `mixed`, hybrid thắng rõ nhất, đạt 100.0%, vì query vừa có từ khóa exact vừa có ý paraphrase; RRF đẩy các document được cả hai retriever đồng ý lên cao. Với `paraphrase`, trong Lite path này BM25 lại nhỉnh hơn hybrid/vector (33.3% vs 32.0%/24.0%) vì embedding `bge-small-en-v1.5` chưa tối ưu cho paraphrase tiếng Việt; nếu dùng model đa ngôn ngữ hơn như `bge-m3`, tôi kỳ vọng vector sẽ thắng slice này.

Tôi không dùng hybrid khi bài toán cần exact match, dễ audit, latency/cost rất thấp, hoặc query là mã lỗi, SKU, log field rõ ràng thì pure BM25 hợp lý hơn. Ngược lại, nếu người dùng hỏi mở, nhiều paraphrase, cross-lingual, và có embedding phù hợp domain, pure vector có thể đủ.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều bất ngờ là hybrid không phải lúc nào cũng thắng từng slice; nó thắng vì ổn định trên nhiều kiểu query. Model embedding và ngôn ngữ của corpus ảnh hưởng rất lớn đến kết quả paraphrase.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: Không
