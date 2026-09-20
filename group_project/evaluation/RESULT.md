# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Ragas 0.4.3 / LangChain 0.4.1 / Python 3.12 |
| Evaluator model                    | nvidia/nemotron-3.5-lightning-30b-a3b |
| Generator model                    | nvidia/nemotron-3.5-lightning-30b-a3b |
| Embedding model                    | BAAI/bge-m3 (dim: 1024) |
| Corpus version/commit              | Commit 738db13 (Branch vinh) |
| Golden dataset size                | 20 ground-truth Q&A cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.3 (calibrated: in-domain cosine 0.65-0.88 vs out-of-domain 0.15-0.28) |

## Configurations

- **Config A — dense-only:** Sử dụng truy vấn ngữ nghĩa thuần từ ChromaDB thông qua vector embedding của mô hình BAAI/bge-m3 với khoảng cách cosine. Trả về top 5 chunks có điểm tương đồng cosine cao nhất.
- **Config B — hybrid + RRF:** Kết hợp truy vấn ngữ nghĩa dense search (top 10) và truy vấn từ khóa BM25Okapi có sàn Lucene IDF (top 10). Tái xếp hạng và hợp nhất bằng giải thuật Reciprocal Rank Fusion (hệ số k=60), trích xuất top 5 chunks tối ưu nhất.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     0.88 |     0.94 |     +0.06 |
| Answer relevance  |     0.86 |     0.92 |     +0.06 |
| Context recall    |     0.79 |     0.91 |     +0.12 |
| Context precision |     0.81 |     0.89 |     +0.08 |
| **Average**       |    0.835 |    0.915 |    +0.080 |

## A/B comparison

- Cấu hình tốt hơn: **Config B (Hybrid + RRF)** vượt trội toàn diện so với Config A ở cả 4 tiêu chí đánh giá, với điểm trung bình tăng từ 0.835 lên 0.915 (+0.080).
- Evidence: Sự khác biệt lớn nhất nằm ở **Context Recall (+0.12)** và **Context Precision (+0.08)**. Đối với các truy vấn chứa từ khóa chuyên môn hẹp, tên nghị định hoặc mã số điều luật (ví dụ: "Nghị định 96/2026/NĐ-CP", "Quyết định 1198/QĐ-TTg", "Khoản 2 Điều 156"), Config A thường xếp các đoạn văn bản chứa đúng từ khóa này ở vị trí rank 6-9 (bị cắt khỏi top 5). Ngược lại, BM25 trong Config B đã kéo chính xác các đoạn này lên đầu, giúp RRF hợp nhất ở vị trí rank 1-2.
- Trade-off về latency/cost: BM25 được tính toán trực tiếp trên RAM với cấu trúc đã cache (thời gian xử lý ~1-2ms), thuật toán RRF chỉ thực hiện tính toán số học trên thứ hạng (<0.5ms). Do đó, Config B hầu như không làm tăng độ trễ (latency tăng không đáng kể dưới 3ms) và hoàn toàn không phát sinh thêm chi phí API so với Config A, trong khi mang lại bước nhảy vọt về chất lượng dữ liệu đầu vào cho generator.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Doanh nghiệp đã được cấp Giấy chứng nhận doanh nghiệp công nghệ cao theo Luật Công nghệ cao 2008 thì có tiếp tục hưởng ưu đãi thuế TNDN không? | Config A |         0.75 |      0.80 |   0.60 |      0.70 | retrieval | Chiến lược chunking cố định 500 ký tự cắt đôi quy định chuyển tiếp: đoạn điều kiện ở chunk trước còn mốc thời gian ở chunk sau, khiến Dense search không gom đủ thông tin trọn vẹn. |
|   2 | Doanh nghiệp sản xuất sản phẩm công nghệ cao khác gì Trung tâm R&D công nghệ cao về mức thuế suất ưu đãi đề xuất? | Config B |         0.80 |      0.85 |   0.85 |      0.80 | generation | Context chứa nhiều con số tỷ lệ phần trăm (10%, 17%) và số năm miễn giảm (4 năm, 2 năm, 9 năm) trong cùng một đoạn văn, khiến LLM có xu hướng tổng hợp nhầm lẫn giữa hai chủ thể nếu không đọc kỹ. |
|   3 | Cơ quan nào trong công ty cổ phần có thẩm quyền bầu, bãi nhiệm và miễn nhiệm Chủ tịch Hội đồng quản trị? | Config A |         0.82 |      0.80 |   0.65 |      0.72 | retrieval | Trùng lặp ngữ nghĩa giữa thẩm quyền của Đại hội đồng cổ đông (bầu thành viên HĐQT) và thẩm quyền của Hội đồng quản trị (bầu Chủ tịch HĐQT), dẫn đến Dense retrieval lấy nhầm điều khoản về ĐHĐCĐ. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Nâng cấp chiến lược chunking sang Hierarchical/Markdown Header-aware Chunking kèm metadata Điều/Khoản | Failure #1 cho thấy chunking theo độ dài ký tự thuần túy làm đứt đoạn các quy định luật có điều kiện và chuyển tiếp phức tạp. | Tăng Context Recall thêm +0.05 và loại bỏ tình trạng đứt mạch ngữ cảnh pháp lý. | Chạy lại unit test đánh giá chunking và so sánh context recall trên các câu hỏi điều khoản chuyển tiếp. |
|        2 | Mở rộng số lượng ứng viên đầu vào cho RRF (Top-15 mỗi nhánh thay vì Top-10) | Failure #3 cho thấy khi gặp các khái niệm gần nghĩa, một số chunk chính xác bị rơi xuống rank 11-12 của Dense search và bị bỏ lỡ trước bước RRF. | Tăng Context Precision và Recall thêm +0.03 trên các truy vấn đa nghĩa. | Đánh giá lại độ phủ kết quả (Recall@K) của danh sách sau khi fuse. |
|        3 | Bổ sung Few-shot CoT (Chain-of-Thought) trong System Prompt cho các bài toán đối chiếu, so sánh số liệu | Failure #2 chứng minh generator đôi khi nhầm lẫn giữa các con số ưu đãi khi hai đối tượng xuất hiện cạnh nhau trong một bảng kê. | Nâng Faithfulness từ 0.94 lên >0.98 cho các câu hỏi so sánh điều kiện. | Đánh giá metric Faithfulness trên tập câu hỏi so sánh đối chiếu số liệu. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Tích hợp Jina Reranker Cloud API (`jina-reranker-v2-base-multilingual`) | Config B (Hybrid + RRF) | Context Precision +0.03, Recall +0.01 | Latency tăng ~180ms (gọi mạng), chi phí API phát sinh | Jina Reranker cho độ chính xác sắp xếp đầu bảng tốt hơn nhẹ ở các câu hỏi so sánh phức tạp, nhưng RRF nội bộ có lợi thế lớn hơn về độ trễ gần như tức thì và hoàn toàn miễn phí. |
