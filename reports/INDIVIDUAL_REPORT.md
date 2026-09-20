# Individual contribution report

## Thông tin

- Họ và tên: Đỗ Lê Việt Anh
- Mã học viên: 2A202602491
- Nhóm: BossNotBot
- Repository/branch: https://github.com/zinhcandoit/K4-L3A-RAG-Pipeline/tree/dlvanh/complete-demo

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 4 — Chunking, embedding, indexing | Đọc Markdown trong `data/standardized/`, chunk bằng `RecursiveCharacterTextSplitter` (size 500, overlap 50), embed bằng `embed_texts()` (hỗ trợ sentence_transformers/openai/gemini), upsert vào ChromaDB (cosine) với ID ổn định `<doc>::chunk-<i>` | `src/task4_chunking_indexing.py` | Done |
| Task 5 — Semantic search | Embed query bằng chính `embed_texts()` của Task 4, đổi cosine distance thành similarity `1 - distance` | `src/task5_semantic_search.py` | Done |
| Task 6 — Lexical search | BM25 (`rank_bm25`) trên cùng corpus chunks, cache index, tokenize bằng `\w+` | `src/task6_lexical_search.py` | Done |
| Task 7 — RRF | `sum(1/(k+rank))`, rank từ 1, khử trùng theo ID, gán `retrieval_method="hybrid"` | `src/task7_reranking.py` | Done |
| Task 8 — PageIndex fallback | Chuyển Markdown sang PDF (SDK chỉ nhận PDF), upload và cache document ID, poll kết quả có timeout, parse thành SearchResult | `src/task8_pageindex_vectorless.py` | Partial — chưa kiểm thử với dịch vụ thật |
| Task 9 — Retrieval pipeline | Dense + BM25, fuse RRF đúng một lần, so threshold với cosine score gốc của dense, fallback PageIndex và trả hybrid nếu fallback lỗi | `src/task9_retrieval_pipeline.py` | Done |
| Task 10 — Generation | `reorder_for_llm`, `format_context` (title + source), `call_llm` (OpenAI/Gemini/Anthropic), safe refusal khi thiếu context hoặc provider lỗi | `src/task10_generation.py` | Done |
| Giao diện Streamlit | Gọi `generate_with_citation`, hiển thị câu trả lời, nguồn, method và score, lưu lịch sử hội thoại, xử lý lỗi | `app.py` | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Dùng `BAAI/bge-m3` (sentence-transformers, chạy local) làm embedding, `normalize_embeddings=True`, dùng chung một hàm `embed_texts()` cho Task 4 và Task 5.  
   **Lý do/evidence:** Corpus là tiếng Việt (văn bản pháp luật và tin tức về hộ kinh doanh); bge-m3 hỗ trợ đa ngôn ngữ, không tốn chi phí API. Dùng chung hàm đảm bảo vector tài liệu và vector query cùng model, cùng số chiều (1024), đúng quy tắc trong `docs/MODULE_CONTRACTS.md`.  
   **Trade-off:** Model nặng (~2 GB tải về, embed chậm trên CPU) nên lần index đầu tiên lâu; đổi lại không phụ thuộc mạng/API khi truy vấn.

2. **Quyết định:** Lọc kết quả BM25 theo token khớp với query thay vì `score > 0`.  
   **Lý do/evidence:** `BM25Okapi` cho IDF = 0 khi một từ xuất hiện ở đúng nửa corpus (ví dụ corpus 2 chunk trong test), khiến mọi score bằng 0 và bị loại hết. Test `test_lexical_search_returns_bm25_contract` fail với `IndexError` trước khi sửa và pass sau khi sửa.  
   **Trade-off:** Có thể giữ lại chunk có score rất thấp (bằng 0); điều này vô hại vì bước RRF ở sau chỉ dùng thứ hạng.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest tests/test_contracts.py -q` (không gọi network/API thật). Query thử thủ công gợi ý: "Từ năm 2026, hộ kinh doanh bị bãi bỏ thuế khoán thì phải nộp thuế theo phương pháp nào và cần chuẩn bị những gì?" và một câu ngoài phạm vi ("Học phí đại học năm 2026 là bao nhiêu?") để kiểm tra threshold/từ chối.
- Kết quả trước/sau nếu có: sau khi hoàn thành Task 4–10, 14/15 test pass; sau khi sửa lỗi BM25, 15/15 test pass.
- Lỗi đã phát hiện và cách xử lý: (a) lỗi BM25 với corpus nhỏ như mô tả ở trên; (b) file Task 8 bị lỗi cú pháp do escape sai đường dẫn font Windows và ký tự xuống dòng, đã sửa và test import lại thành công.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: chưa chạy đánh giá RAG (RAGAS) nên chưa có số liệu faithfulness/recall/precision; `reports/RESULT.md` vẫn là template. Task 8 chưa được thử với PageIndex thật (tên field trong response được đọc từ mã nguồn SDK), và PDF sinh ra từ Markdown chỉ là văn bản thuần, mất cấu trúc heading/bảng. `SCORE_THRESHOLD` đang dùng giá trị mặc định 0.3, chưa hiệu chỉnh bằng query in-domain/out-of-domain.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: xây golden dataset và chạy đánh giá A/B (dense-only và hybrid + RRF) để hiệu chỉnh threshold bằng số liệu thay vì giá trị mặc định.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-20
- Tên thành viên: Đỗ Lê Việt Anh
