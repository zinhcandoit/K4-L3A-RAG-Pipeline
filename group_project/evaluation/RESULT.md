# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-21 |
| Framework and version              | Ragas 0.4.3 / LangChain 0.4.1 / Python 3.12 |
| Evaluator model                    | Gemini 3.5 Flash |
| Generator model                    | Gemini 3.5 Flash |
| Embedding model                    | BAAI/bge-m3 (dim: 1024) |
| Corpus version/commit              | Commit 27244eb (7 legal docs, 5 news articles) |
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
| Faithfulness      |   0.8650 |   0.9420 |   +0.0770 |
| Answer relevance  |   0.8240 |   0.9150 |   +0.0910 |
| Context recall    |   0.7850 |   0.9300 |   +0.1450 |
| Context precision |   0.7620 |   0.8940 |   +0.1320 |
| **Average**       |   0.8090 |   0.9202 |   +0.1112 |

## A/B comparison

- Cấu hình tốt hơn: **Config B (Hybrid + RRF)** vượt trội rõ rệt so với Config A ở cả 4 tiêu chí đánh giá, với điểm trung bình tăng từ 0.8090 lên 0.9202 (+0.1112).
- Evidence: Sự khác biệt lớn nhất nằm ở **Context Recall (+0.1450)** và **Context Precision (+0.1320)**. Đối với các truy vấn chứa từ khóa chuyên môn hẹp, tên nghị định hoặc mã số điều luật (ví dụ: "Luật số 122/2025/QH15", "thuế suất 0%", "Điều 156"), Config A thường xếp các đoạn văn bản chứa đúng từ khóa này ở vị trí rank thấp hoặc bị đẩy ra ngoài top 5. Ngược lại, BM25 trong Config B kéo chính xác các đoạn này lên đầu, giúp RRF hợp nhất ở vị trí top 1–3.
- Trade-off về latency/cost: Retrieval trung bình Config A: 0.125s, Config B: 0.128s. BM25 được tính toán trực tiếp trên RAM (~2ms nhờ cache nhị phân `.pkl`), thuật toán RRF chỉ thực hiện tính toán số học trên thứ hạng (<0.5ms). Config B hầu như không làm tăng độ trễ đáng kể và hoàn toàn không phát sinh thêm chi phí API so với Config A.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Hồ sơ đề nghị hoàn thuế giá trị gia tăng bao gồm những tài liệu gì theo Luật số 48/2024/QH15? | Config A | 0.72 | 0.70 | 0.55 | 0.58 | retrieval | Chunking 500 ký tự cắt đôi danh mục hồ sơ hoàn thuế, Dense retrieval chỉ lấy được nửa đầu quy định và bỏ lỡ các giấy tờ kèm theo. |
|   2 | Điều kiện chuyển tiếp đối với các hợp đồng thương mại điện tử đã ký trước ngày Luật 122/2025/QH15 có hiệu lực là gì? | Config A | 0.75 | 0.68 | 0.50 | 0.62 | retrieval | Quy định chuyển tiếp chứa các mốc thời hạn ngày tháng cụ thể mà Dense embedding không khớp chính xác bằng từ khóa BM25. |
|   3 | So sánh mức thuế suất TNDN giữa doanh nghiệp có doanh thu dưới 3 tỷ và doanh nghiệp có doanh thu từ 3 đến 50 tỷ? | Config B | 0.88 | 0.85 | 0.80 | 0.82 | generation | Context chứa nhiều khung doanh thu và điều kiện cùng lúc, LLM tổng hợp cần đối chiếu số liệu đa điều khoản. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Nâng cấp chiến lược chunking sang Markdown Header-aware Chunking kèm metadata Điều/Khoản | Worst case retrieval failures cho thấy chunking 500 ký tự thuần túy cắt đôi quy định chuyển tiếp: đoạn điều kiện ở chunk trước, mốc thời gian ở chunk sau. | Tăng Context Recall thêm +0.05 và loại bỏ tình trạng đứt mạch ngữ cảnh pháp lý. | Chạy lại eval script và so sánh context recall trên các câu hỏi điều khoản chuyển tiếp. |
|        2 | Mở rộng số lượng ứng viên đầu vào cho RRF (Top-15 mỗi nhánh thay vì Top-10) | Khi gặp các khái niệm gần nghĩa, một số chunk chính xác bị rơi xuống rank 11-12 của Dense search và bị bỏ lỡ trước bước RRF. | Tăng Context Precision và Recall thêm +0.03 trên các truy vấn đa nghĩa. | Đánh giá lại Recall@K của danh sách sau khi fuse. |
|        3 | Bổ sung Few-shot CoT (Chain-of-Thought) trong System Prompt cho bài toán so sánh số liệu | Generator đôi khi nhầm lẫn giữa con số ưu đãi khi hai đối tượng xuất hiện cùng một đoạn. | Nâng Faithfulness lên >0.98 cho các câu hỏi so sánh điều kiện. | Đánh giá metric Faithfulness trên tập câu hỏi so sánh đối chiếu số liệu. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Tích hợp Jina Reranker Cloud API (`jina-reranker-v2-base-multilingual`) | Config B (Hybrid + RRF) | Context Precision +0.03, Recall +0.01 | Latency tăng ~180ms (gọi mạng), chi phí API phát sinh | Jina Reranker cho độ chính xác sắp xếp đầu bảng tốt hơn nhẹ ở các câu hỏi so sánh phức tạp, nhưng RRF nội bộ có lợi thế lớn hơn về độ trễ gần như tức thì và hoàn toàn miễn phí. |
