# Individual contribution report

Mỗi thành viên copy template này thành:

```text
reports/<student-id>-<short-name>.md
```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---

## Thông tin

- Họ và tên: Thiều Quang Vinh
- Mã học viên: 2A202602877
- Nhóm: BossNotBot
- Repository/branch: https://github.com/zinhcandoit/K4-L3A-RAG-Pipeline.git (branch: `vinh`)

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Thu thập & Chuẩn hóa dữ liệu (Task 1, 2, 3) | Thu thập 7 tài liệu văn bản luật (.doc, .docx) và crawl 5 bài viết chính sách. Xây dựng parser nhị phân OLE CFBF/Piece Table trích xuất file .doc Word 97-2004, kết hợp MarkItDown/python-docx cho .docx và chuẩn hóa làm sạch Markdown. | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `src/task3_convert_markdown.py` (commit `9bc3a2e`, `b838352`, `c02ce7a`, `d57fa4a`) | Done |
| Chunking & Indexing ChromaDB (Task 4, 5) | Cấu hình chia đoạn văn bản bằng `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50), nhúng vector bằng mô hình đa ngữ `BAAI/bge-m3` (dim=1024), lưu trữ và index vào ChromaDB với metric cosine distance (`hnsw:space: cosine`). | `src/task4_chunking_indexing.py`, `src/task5_semantic_search.py` (commit `738db13`, `e0d11da`, `5faf4a0`) | Done |
| Lexical Search & BM25 Caching (Task 6) | Xây dựng BM25 Search với class `BM25OkapiWithFloor` kế thừa bổ sung sàn Lucene-style non-negative IDF cho corpus nhỏ tiếng Việt; tối ưu hóa serialize index và corpus ra đĩa nhị phân `data/bm25_cache.pkl` nạp tức thì trong 2ms. | `src/task6_lexical_search.py` (commit `738db13`, `a4b42cc`, `e0d11da`) | Done |
| Reranking & Retrieval Pipeline (Task 7, 8, 9) | Triển khai giải thuật Reciprocal Rank Fusion (RRF, $k=60$) hợp nhất thứ hạng Dense Search và Lexical Search; tích hợp thêm Jina AI Reranker API (`jina-reranker-v2-base-multilingual`) để đối sánh; xây dựng pipeline fallback PageIndex khi dense cosine score < 0.3. | `src/task7_reranking.py`, `src/task8_pageindex_vectorless.py`, `src/task9_retrieval_pipeline.py` (commit `738db13`, `a4b42cc`) | Done |
| Generation with Citations & Context Reorder (Task 10) | Xây dựng pipeline sinh câu trả lời kèm citation `[1]`, `[2]`, áp dụng kỹ thuật `reorder_for_llm` giảm hiện tượng lost-in-the-middle, kết nối LLM (OpenAI/NVIDIA NIM Nemotron-3.5, Gemini) và cơ chế safe refusal khi dữ liệu không đủ căn cứ. | `src/task10_generation.py` (commit `738db13`, `a4b42cc`, `5faf4a0`, `5f8422d`) | Done |
| Giao diện ứng dụng Streamlit (UI/Demo) | Phát triển giao diện Web tương tác Streamlit: chat UI, tùy chọn chuyển đổi mô hình Rerank (RRF vs Jina), tùy chỉnh Top-K, hiển thị chi tiết trích dẫn kèm score và phương thức truy vấn; nạp sẵn model & BM25 vào RAM qua `@st.cache_resource`. | `app.py` (commit `738db13`, `a4b42cc`, `e0d11da`) | Done |
| Đánh giá A/B Benchmark & Ragas Eval | Xây dựng bộ Golden Dataset (20 câu hỏi ground-truth) và kịch bản `run_eval.py` thực hiện đánh giá A/B giữa Config A (dense-only) và Config B (hybrid + RRF) đo 4 metric: Faithfulness, Answer Relevance, Context Recall, Context Precision. | `group_project/evaluation/run_eval.py`, `golden_dataset.json`, `RESULT.md` (commit `d57fa4a`, `3d712a0`) | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Áp dụng Hybrid Retrieval kết hợp Dense Search (`BAAI/bge-m3`) và Lexical Search (`BM25Okapi` có Lucene IDF floor) dung hợp qua Reciprocal Rank Fusion (RRF, $k=60$) làm cấu hình mặc định (Config B) thay vì chỉ dùng Dense Search thuần túy (Config A).  
   **Lý do/evidence:** Trên tập đánh giá 20 câu hỏi pháp lý thực tế trong `RESULT.md`, Config A bộc lộ điểm yếu khi xử lý các từ khóa chính xác như số hiệu văn bản ("Luật số 122/2025/QH15", "Điều 156"), thuật ngữ chuyên ngành hẹp ("thuế suất 0%") khiến tài liệu liên quan bị rơi ra khỏi top 5, dẫn đến Context Recall chỉ đạt 0.7850 và Context Precision 0.7620. Khi áp dụng Config B, BM25 kéo chính xác các chunk chứa từ khóa lên đầu, RRF hợp nhất ở rank 1-3, giúp Context Recall tăng vượt bậc lên 0.9300 (+0.1450), Context Precision đạt 0.8940 (+0.1320), Faithfulness đạt 0.9420 (+0.0770) và điểm trung bình tăng từ 0.8090 lên 0.9202 (+0.1112).  
   **Trade-off:** Cần duy trì song song 2 chỉ mục và thực hiện 2 lần truy vấn trước khi gộp. Để giải quyết, tôi đã tối ưu hóa lưu BM25 index ra cache nhị phân `.pkl` nạp lên RAM chỉ mất ~2ms, thuật toán RRF tính toán số học trên thứ hạng (<0.5ms). Nhờ vậy, độ trễ chỉ tăng nhẹ từ 0.125s lên 0.128s và hoàn toàn không phát sinh thêm chi phí API bên ngoài.

2. **Quyết định:** Sử dụng điểm Best Dense Cosine Score (ngưỡng 0.3 đã cân chỉnh) để kích hoạt cơ chế Fallback sang PageIndex Vectorless Search, thay vì sử dụng điểm RRF sau khi fuse.  
   **Lý do/evidence:** Điểm RRF $1/(k + rank)$ chỉ phản ánh thứ hạng tương đối trong tập ứng viên trả về của một truy vấn cụ thể, không đại diện cho độ tương đồng ngữ nghĩa tuyệt đối của câu hỏi với kho dữ liệu. Trong khi đó, cosine score của mô hình `bge-m3` đã được kiểm chứng có ranh giới rõ rệt: câu hỏi in-domain đạt cosine từ 0.65 - 0.88, còn out-of-domain chỉ đạt 0.15 - 0.28. Việc dựa trên best dense score đảm bảo phân biệt chính xác khi nào truy vấn nằm ngoài miền dữ liệu để fallback hợp lý mà không kích hoạt nhầm.  
   **Trade-off:** Khi truy vấn dense gặp lỗi hoặc score biên (xung quanh 0.3), hệ thống phải bọc thêm khối try-catch bảo vệ để nếu fallback PageIndex thất bại vẫn trả kết quả hybrid an toàn thay vì gây gián đoạn pipeline.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng:
  - Bộ test suites tự động trong `tests/test_acceptance.py` và `tests/test_contracts.py` kiểm tra toàn bộ điều kiện nghiệm thu: định dạng tài liệu landing, tính hợp lệ của Markdown standardized, hợp đồng dữ liệu giữa các module và 20 ground-truth cases.
  - Bộ 20 câu hỏi ground-truth trong `group_project/evaluation/golden_dataset.json` bao phủ đa dạng tình huống pháp lý: truy vấn điều khoản cụ thể, đối chiếu số liệu (khung thuế suất TNDN theo doanh thu) và các điều kiện chuyển tiếp.
  - Script đánh giá A/B thực nghiệm `group_project/evaluation/run_eval.py` đo 4 metrics Ragas chuẩn LLM-as-a-judge.
- Kết quả trước/sau nếu có:
  - Cấu hình A (Dense-only baseline): Faithfulness: 0.8650 | Answer relevance: 0.8240 | Context recall: 0.7850 | Context precision: 0.7620 | Điểm trung bình: 0.8090 | Latency retrieval: 0.125s.
  - Cấu hình B (Hybrid + RRF): Faithfulness: 0.9420 (+0.0770) | Answer relevance: 0.9150 (+0.0910) | Context recall: 0.9300 (+0.1450) | Context precision: 0.8940 (+0.1320) | Điểm trung bình: 0.9202 (+0.1112) | Latency retrieval: 0.128s.
- Lỗi đã phát hiện và cách xử lý:
  - *BM25 IDF âm trên tập tài liệu nhỏ:* `BM25Okapi` gốc gán điểm IDF âm cho các thuật ngữ xuất hiện ở nhiều chunk, làm đảo lộn thứ tự ưu tiên. Cách xử lý: kế thừa `BM25OkapiWithFloor` triển khai sàn Lucene-style non-negative IDF trong `src/task6_lexical_search.py`.
  - *Độ trễ khởi động khi nạp lại BM25:* Phải tokenize toàn bộ corpus mỗi lần chạy. Cách xử lý: lưu index ra file nhị phân `data/bm25_cache.pkl` nạp trong ~2ms và lưu cache trên RAM qua `@st.cache_resource` trong `app.py`.
  - *Lỗi đọc file Word .doc cũ (97-2004):* Thư viện thông thường không đọc được định dạng nhị phân OLE CFBF. Cách xử lý: tự xây dựng hàm `_extract_doc` bóc tách luồng Piece Table MS-DOC FIB trong `src/task3_convert_markdown.py`.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Chiến lược chunking hiện tại dùng `RecursiveCharacterTextSplitter` cố định (500 ký tự, overlap 50) ngắt theo dòng và ký tự thuần túy, chưa nhận biết cấu trúc Điều/Khoản phân cấp của văn bản luật. Hệ quả là một số danh mục quy định dài (như danh mục hồ sơ hoàn thuế tại Luật 48/2024/QH15) bị chia cắt đôi, khiến retriever chỉ lấy được nửa đầu và bỏ lỡ các giấy tờ ở nửa sau.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Nâng cấp sang Markdown Header-aware Chunking theo cấu trúc phân cấp pháp lý (Chương > Mục > Điều > Khoản), đồng thời bổ sung metadata số hiệu Điều vào từng chunk để retriever không bao giờ cắt vụn ngữ cảnh của một điều luật.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 21/09/2026
- Tên thành viên: Thiều Quang Vinh
