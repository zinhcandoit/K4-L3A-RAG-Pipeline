# RAG evaluation results

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-20 |
| Framework and version | RAGAS 0.4.3 |
| Evaluator model | `gpt-5.4-mini` |
| Generator model | `gpt-5.6-luna` |
| Embedding model | `text-embedding-3-small` |
| Corpus version/commit | `bff0681` — 5 văn bản luật + 10 bài báo, 1163 chunk |
| Golden dataset size | 16 case, mỗi case có anchor trích nguyên văn đã kiểm chứng tồn tại trong corpus |
| `top_k` | 5 |
| Fallback threshold and calibration | 0.45 — hiệu chỉnh bằng in-domain "hộ kinh doanh nộp thuế 2026" (best dense 0.7930), out-of-domain "chăm sóc lan hồ điệp" (0.3804) và "Messi World Cup 2022" (0.2193) |

## Configurations

- **Config A — dense-only:** `retrieve(use_reranking=False)` — chỉ dense search trên ChromaDB cosine, lấy `dense[:top_k]`. Không gọi BM25.
- **Config B — hybrid + RRF:** `retrieve(use_reranking=True)` — dense + BM25, hợp nhất bằng RRF (`k=60`) đúng một lần.

Hai config dùng chung golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.7969 | 0.7344 | -0.0625 |
| Answer relevance | 0.6507 | 0.7030 | +0.0523 |
| Context recall | 0.6875 | 0.8125 | +0.1250 |
| Context precision | 0.7024 | 0.7830 | +0.0806 |
| **Average** | 0.7094 | 0.7582 | +0.0488 |

## A/B comparison

- Cấu hình tốt hơn: **Config B — hybrid + RRF**
- Evidence: Config B thắng ở 3/4 metric, mạnh nhất ở context recall (+0.1250) và context precision (+0.0806). Đúng với kỳ vọng: BM25 bắt được số hiệu văn bản và thuật ngữ chính xác ("01/TKN-CNKD", "500 triệu đồng", "68/2026/NĐ-CP") mà dense hay trượt, còn RRF hợp nhất theo thứ hạng nên chunk mạnh ở cả hai phía được đẩy lên. Ngược lại faithfulness giảm (-0.0625): kéo thêm chunk lexical vào top-k làm context rộng hơn, model có nhiều chỗ để suy diễn hơn. Đổi lại recall cao hơn, nên trung bình vẫn nghiêng về B.
- Trade-off về latency/cost: retrieval trung bình 0.9833s (A) so với 0.6032s (B), chênh -0.3801s; generation 2.8797s so với 2.9790s. **Không kết luận B nhanh hơn A từ con số này.** Cả hai config đều phải gọi API `text-embedding-3-small` để embed query, và lượt gọi mạng đó lấn át hoàn toàn phần tính toán cục bộ; BM25 trên 1163 chunk chạy trong bộ nhớ nên gần như không đo được bên cạnh độ nhiễu của API. Muốn so latency retrieval cho nghiêm thì phải tách riêng thời gian embed query, chạy nhiều vòng và lấy trung vị. Về chi phí token thì hai config bằng nhau: B không thêm lượt gọi LLM nào, chỉ thêm một lượt BM25 cục bộ.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Khi Giấy chứng nhận đăng ký hộ kinh doanh ghi sai so với hồ sơ thì bao lâu được cấp lại? | B | 0.3333 | 0.0000 | 0.0000 | 0.0000 | retrieval (chunk miss) | Lấy đúng tài liệu `nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh.md` nhưng trượt chunk chứa câu trả lời (context recall 0.00). Model từ chối thay vì bịa — hành vi đúng, nhưng RAGAS chấm answer relevance = 0 cho câu từ chối nên điểm tụt sâu. |
| 2 | Ngưỡng doanh thu năm nào được Nghị định 68/2026 dùng để quy định riêng việc khai thuế? | B | 0.5000 | 0.0000 | 0.0000 | 0.0000 | retrieval (chunk miss) | Lấy đúng tài liệu `nd-68-2026-chinh-sach-thue-ho-kinh-doanh.md` nhưng trượt chunk chứa câu trả lời (context recall 0.00). Model từ chối thay vì bịa — hành vi đúng, nhưng RAGAS chấm answer relevance = 0 cho câu từ chối nên điểm tụt sâu. |
| 3 | Hồ sơ đăng ký thành lập hộ kinh doanh gồm những giấy tờ gì? | B | 0.2500 | 0.0000 | 0.0000 | 1.0000 | retrieval (chunk miss) | Lấy đúng tài liệu `nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh.md` nhưng trượt chunk chứa câu trả lời (context recall 0.00). Model từ chối thay vì bịa — hành vi đúng, nhưng RAGAS chấm answer relevance = 0 cho câu từ chối nên điểm tụt sâu. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Cắt NĐ 168/2025 chỉ giữ Chương VIII (hộ kinh doanh) | Một văn bản chiếm 889/1163 chunk (76,4%); khoảng 30-70% file không nhắc "hộ kinh doanh" lần nào | Giảm nhiễu, tăng context precision cho câu hỏi về thuế | Chạy lại `run_evaluation.py`, so context precision trước/sau |
| 2 | Tìm nguồn toàn văn thật cho NĐ 68/2026 | Trang "toàn văn" hiện tại chỉ là tóm tắt 2.587 ký tự, ra 9 chunk (0,8%) trong khi đây là nghị định thuế trọng tâm | Tăng context recall cho nhóm câu hỏi về thuế | Đếm lại chunk theo tài liệu, chạy lại A/B |
| 3 | Tăng `CHUNK_SIZE` cho tài liệu luật (500 → 1000-1200) hoặc chunk theo ranh giới Điều | 3/3 case kém nhất là *chunk miss*: lấy đúng tài liệu nhưng trượt đúng đoạn chứa câu trả lời, ví dụ "03 ngày làm việc" và "trên 500 triệu đồng" | Tăng context recall; giảm số câu bị từ chối oan | Chạy lại `run_evaluation.py`, so context recall và đếm lại số câu trả lời là REFUSAL |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| Chưa thực hiện | — | — | — | Nhóm chưa làm phần bonus; theo rubric bonus chỉ tính khi có baseline, metric delta và thay đổi latency/cost |
