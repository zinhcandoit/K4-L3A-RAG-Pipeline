Sau khi đọc kỹ toàn bộ tài liệu trong repo (gồm các file Markdown [README.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/README.md), [STEP_BY_STEP.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/docs/STEP_BY_STEP.md), [MODULE_CONTRACTS.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/docs/MODULE_CONTRACTS.md), [GRADING_RUBRIC.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/docs/GRADING_RUBRIC.md), [RESULT.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/group_project/evaluation/RESULT.md)) cùng 6 file HTML hướng dẫn chi tiết từ web VLearn (`day8-1.html` → `day8-6.html`) và hệ thống test có sẵn ([test_contracts.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/tests/test_contracts.py), [test_acceptance.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/tests/test_acceptance.py)), dưới đây là **hướng dẫn tường minh từng bước thực hiện bài Lab** được điều chỉnh hoàn toàn cho công cụ **`uv`** trên môi trường Windows.

---

# 🗺️ Tổng Quan Kiến Trúc & Yêu Cầu Cốt Lõi

1. **Mục tiêu**: Xây dựng chatbot RAG hoàn chỉnh trên dữ liệu tự thu thập (≥3 văn bản chính sách/quy định + ≥5 bài viết/tin tức).
2. **Pipeline**: Thu thập dữ liệu → Chuẩn hóa Markdown → Chunking & Embedding → Hybrid Retrieval (Dense ChromaDB + BM25) → Reciprocal Rank Fusion (RRF) → Fallback mechanism (PageIndex/Safe Refusal) → Generation có Citation → Streamlit UI → Đánh giá A/B (Dense-only vs Hybrid) trên Golden Dataset (≥15 câu).
3. **Các quy tắc sống còn (Contract Invariants)**:
   - **Task 4 và Task 5** phải dùng chung hàm `embed_texts()` (cùng model, dimension, không lệch không gian vector).
   - **RRF**: Chỉ gộp thứ hạng, tính theo công thức $RRF(d) = \sum \frac{1}{k + rank}$ ($rank \ge 1$), **không sửa (mutate)** trực tiếp item gốc trong danh sách đầu vào, kết quả có `retrieval_method="hybrid"`.
   - **Fallback**: Quyết định fallback **bắt buộc so sánh ngưỡng threshold với cosine score gốc của dense search**, tuyệt đối không dùng điểm RRF.
   - **Safe Refusal**: Khi thiếu bằng chứng hoặc dưới threshold mà fallback thất bại, LLM phải trả câu từ chối an toàn với `sources=[]` và `retrieval_source="none"`.

---

# 🚀 Hướng Dẫn Chi Tiết Từng Bước Thực Hiện

## Bước 1: Khởi Tạo Môi Trường Bằng `uv` & Cấu Hình Nhóm
*(Tương ứng tài liệu VLearn Bài 1 & [README.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/README.md))*

### 1.1. Cài đặt môi trường bằng `uv`
Trong terminal PowerShell tại thư mục dự án:
```powershell
# 1. Tạo môi trường ảo với uv (khuyến nghị Python 3.10 - 3.12 theo pyproject.toml)
uv venv --python 3.11

# 2. Kích hoạt môi trường ảo
.venv\Scripts\activate

# 3. Cài đặt toàn bộ dependencies trong pyproject.toml và dev dependencies
uv pip install -e ".[dev]"

# 4. Cài đặt Chromium cho Playwright (phục vụ crawl dữ liệu bằng Crawl4AI)
uv run playwright install chromium
```

### 1.2. Tạo file cấu hình `.env`
Sao chép `.env.example` thành `.env`:
```powershell
Copy-Item .env.example .env
```
Mở `.env` và cấu hình các biến phù hợp:
- `LLM_PROVIDER`: Chọn `gemini`, `openai` hoặc `anthropic`.
- API Key tương ứng: `GEMINI_API_KEY`, `OPENAI_API_KEY`,...
- `EMBEDDING_PROVIDER`: Chọn `sentence_transformers` (chạy local, model `BAAI/bge-m3`) hoặc `openai`/`gemini`.
- `SCORE_THRESHOLD`: Ban đầu để trống hoặc đặt tạm `0.3` (sẽ hiệu chỉnh sau ở Bước 4).

### 1.3. Khai báo thông tin nhóm
Tạo file `TEAMMATES.md` tại thư mục gốc của repo với nội dung:
```markdown
# Phân công nhóm

- Tên nhóm: [Tên nhóm]
- Đề tài: [Chủ đề lựa chọn]

| Họ và tên | Mã sinh viên | Vai trò | Phần việc phụ trách |
|---|---|---|---|
| Nguyễn Văn A | ... | Leader / Data | Task 1, 2, 3 |
| Trần Thị B | ... | Retrieval | Task 4, 5, 6, 7 |
| Lê Văn C | ... | Generation & UI | Task 8, 9, 10, app.py |
| Phạm Văn D | ... | Evaluation | Golden dataset, RESULT.md |
```

---

## Bước 2: Thu Thập & Chuẩn Hóa Corpus Dữ Liệu
*(Tương ứng tài liệu VLearn Bài 2 & Task 1, 2, 3)*

Mỗi nhóm chọn một chủ đề (gợi ý trong [SUGGESTED_TOPICS.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/docs/SUGGESTED_TOPICS.md), ví dụ: *Tuyển sinh Đại học*, *Dịch vụ sinh viên*, *Chính sách đào tạo*).

### 2.1. Thu thập văn bản chính sách (Task 1)
- Tải tối thiểu **3 file** PDF hoặc DOCX (kích thước mỗi file > 1KB) lưu vào:
  `data/landing/legal/` (ví dụ: `hoc-phi.pdf`, `hoc-bong.pdf`, `ky-tuc-xa.pdf`).
- Có thể tải thủ công hoặc hoàn thiện script [src/task1_collect_legal_docs.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task1_collect_legal_docs.py) dùng `requests` để tải tự động.

### 2.2. Crawl bài viết/tin tức (Task 2)
- Mở [src/task2_crawl_news.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task2_crawl_news.py):
  - Thêm ít nhất 5 URL công khai vào `ARTICLE_URLS`.
  - Hoàn thiện hàm `crawl_article(url)` sử dụng `crawl4ai` (hoặc `requests + BeautifulSoup`).
  - Mỗi file xuất ra `data/landing/news/article_XX.json` phải có đủ 4 trường:
    ```json
    {
      "url": "https://...",
      "title": "Tiêu đề bài viết",
      "date_crawled": "2026-09-20T...",
      "content_markdown": "# Nội dung..."
    }
    ```
- Chạy lệnh thu thập:
  ```powershell
  uv run python -m src.task2_crawl_news
  ```

### 2.3. Chuẩn hóa sang Markdown (Task 3)
- Mở [src/task3_convert_markdown.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task3_convert_markdown.py):
  - `convert_legal_docs()`: Dùng thư viện `markitdown` (hoặc `pypdf`/`pymupdf`) chuyển đổi các file trong `data/landing/legal/` thành các file `.md` trong `data/standardized/legal/`.
  - `convert_news_articles()`: Đọc các file JSON trong `data/landing/news/`, ghép phần header metadata (`# Title`, `Source`, `Crawled`) với `content_markdown`, lưu thành các file `.md` trong `data/standardized/news/`.
- Chạy lệnh chuẩn hóa:
  ```powershell
  uv run python -m src.task3_convert_markdown
  ```
- **Kiểm tra tiêu chí chấp nhận (Acceptance Check)**:
  ```powershell
  uv run pytest tests/test_acceptance.py -k "corpus or standardized" -q
  ```
  *(Phải pass 3 bài test: `test_corpus_has_required_legal_documents`, `test_corpus_has_required_news_with_metadata`, `test_standardized_output_covers_both_source_types`)*.

---

## Bước 3: Chunking, Indexing Vào ChromaDB & BM25
*(Tương ứng tài liệu VLearn Bài 3 & Task 4, 5, 6)*

### 3.1. Chunking & Indexing (Task 4)
Mở [src/task4_chunking_indexing.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task4_chunking_indexing.py) và hoàn thiện các hàm theo [MODULE_CONTRACTS.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/docs/MODULE_CONTRACTS.md):
- `load_documents()`: Đọc tất cả file `.md` trong `data/standardized/`. Gán ID ổn định (dựa trên đường dẫn tương đối), metadata gồm `source`, `title`, `doc_type` ("legal" hoặc "news"), `url`.
- `chunk_documents()`: Dùng `RecursiveCharacterTextSplitter` với `chunk_size=500`, `chunk_overlap=50`. ID của chunk phải theo định dạng `{doc_id}::chunk-{index}`, metadata bổ sung thêm `chunk_index`.
- `embed_texts(texts)`: Dùng `SentenceTransformer("BAAI/bge-m3")` (hoặc provider trong `.env`) để trả về `list[list[float]]`.
- `embed_chunks(chunks)`: Gọi `embed_texts()` trên toàn bộ chunk content, gán vector vào `chunk["embedding"]`.
- `get_collection()`: Mở hoặc tạo ChromaDB collection với `metadata={"hnsw:space": "cosine"}`.
- `index_to_vectorstore(chunks)`: Thực hiện `upsert` vào collection. Dùng upsert với ID cố định đảm bảo chạy lại nhiều lần không bị duplicate dữ liệu.
- Chạy index:
  ```powershell
  uv run python -m src.task4_chunking_indexing
  ```

### 3.2. Semantic Search (Task 5)
Mở [src/task5_semantic_search.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task5_semantic_search.py):
- Hàm `semantic_search(query, top_k=10)`:
  - Embed query bằng đúng hàm `embed_texts([query])[0]`.
  - Gọi `get_collection().query(...)`.
  - **Lưu ý chuyển đổi score**: Vì ChromaDB dùng cosine distance $d \in [0, 2]$, chuyển sang cosine similarity bằng `score = max(0.0, 1.0 - distance)`.
  - Trả về danh sách `SearchResult` có `retrieval_method="dense"`, sắp xếp theo `score` giảm dần, tối đa `top_k`.

### 3.3. Lexical Search với BM25 (Task 6)
Mở [src/task6_lexical_search.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task6_lexical_search.py):
- Xây dựng BM25 index trên **cùng tập corpus chunks** của Task 4.
- Hàm `build_bm25_index(corpus)`: Tách từ văn bản `item["content"].lower().split()` và khởi tạo `BM25Okapi`.
- Hàm `lexical_search(query, top_k=10)`: Tính điểm BM25 cho query, lọc các score > 0, sắp xếp giảm dần, trả về danh sách `SearchResult` có `retrieval_method="bm25"`.

### 3.4. Kiểm tra Contract Phase Retrieval
```powershell
uv run pytest tests/test_contracts.py -k "chunk or semantic or lexical" -q
```

---

## Bước 4: Reciprocal Rank Fusion (RRF) & Cơ Chế Fallback
*(Tương ứng tài liệu VLearn Bài 4 & Task 7, 8, 9)*

### 4.1. Reciprocal Rank Fusion (Task 7)
Mở [src/task7_reranking.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task7_reranking.py):
- Hàm `rerank_rrf(ranked_lists, top_k=5, k=60)`:
  - Duyệt qua từng danh sách, với mỗi item ở vị trí `rank` (bắt đầu từ 1):
    $$\text{score}[id] = \sum \frac{1}{k + rank}$$
  - **Không mutate item đầu vào**: Tạo bản copy của item trước khi cập nhật `score` và `retrieval_method="hybrid"`.
  - Trả về kết quả sắp xếp theo RRF score giảm dần, không trùng lặp ID, tối đa `top_k`.

### 4.2. Vectorless Fallback (Task 8)
Mở [src/task8_pageindex_vectorless.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task8_pageindex_vectorless.py):
- Triển khai `pageindex_search(query, top_k=5)`.
- Nếu không sử dụng dịch vụ PageIndex từ bên thứ 3 (hoặc không có API key), cấu hình hàm xử lý ngoại lệ an toàn hoặc trả về rỗng để Task 9 xử lý fallback graceful mà không bị crash ứng dụng.

### 4.3. Pipeline Tích Hợp & Hiệu Chỉnh Threshold (Task 9)
Mở [src/task9_retrieval_pipeline.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task9_retrieval_pipeline.py):
- Hàm `retrieve(query, top_k=5, score_threshold=0.3, use_reranking=True)`:
  1. Lấy kết quả từ `semantic_search(query, top_k=top_k*2)` và `lexical_search(query, top_k=top_k*2)`.
  2. Lấy **cosine score gốc cao nhất** từ `dense` results (`dense[0]["score"]` nếu có).
  3. Nếu `best_dense_score < score_threshold`: thử gọi `pageindex_search(query, top_k=top_k)`.
     - Nếu `pageindex_search` thành công và có kết quả: trả về kết quả đó (method là `"pageindex"`).
     - Nếu `pageindex_search` phát sinh lỗi hoặc không có kết quả: **không để crash**, quay lại fallback trả về hybrid/dense results.
  4. Nếu `best_dense_score >= score_threshold` (hoặc sau khi fallback lỗi):
     - Nếu `use_reranking=True`: gọi `rerank_rrf([dense, sparse], top_k=top_k)`.
     - Nếu `use_reranking=False` (dùng cho Config A khi đánh giá): chỉ lấy `dense[:top_k]`.
- **Hiệu chỉnh SCORE_THRESHOLD**:
  - Thử 1 câu hỏi **In-Domain** (ví dụ: *"Học phí kỳ 1 là bao nhiêu?"*) → xem best dense score (ví dụ: `0.72`).
  - Thử 1 câu hỏi **Out-of-Domain** (ví dụ: *"Cách làm bánh chưng ngày tết?"*) → xem best dense score (ví dụ: `0.18`).
  - Chọn `SCORE_THRESHOLD` ở khoảng giữa (ví dụ: `0.35` hoặc `0.40`) và lưu vào `.env`.
- Chạy kiểm tra:
  ```powershell
  uv run pytest tests/test_contracts.py -k "rrf or retrieve" -q
  ```

---

## Bước 5: Generation Có Citation & Xây Dựng Giao Diện Streamlit
*(Tương ứng tài liệu VLearn Bài 5 & Task 10, app.py)*

### 5.1. Generation Có Citation (Task 10)
Mở [src/task10_generation.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/src/task10_generation.py):
- `reorder_for_llm(chunks)`: Sắp xếp lại thứ tự chunks để tránh hiện tượng *lost-in-the-middle* (đưa các chunk có score cao nhất về đầu và cuối context), **không làm mất hay sửa ID**.
- `format_context(chunks)`: Format các chunk thành chuỗi văn bản có gắn rõ nhãn: `[Document X | Title: ... | Source: ...]`.
- `call_llm(system_prompt, user_message)`: Gọi API tương ứng với `LLM_PROVIDER` cấu hình trong `.env` (`gemini`, `openai`, hoặc `anthropic`), trả về văn bản text thuần.
- `generate_with_citation(query, top_k=5)`:
  - Gọi `chunks = retrieve(query, top_k=top_k)`.
  - **Xử lý Safe Refusal**: Nếu `chunks` rỗng hoặc không có evidence phù hợp:
    ```python
    return {
        "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
        "sources": [],
        "retrieval_source": "none"
    }
    ```
  - Nếu có chunks: đưa qua `reorder_for_llm` → `format_context` → gọi `call_llm` với prompt yêu cầu bắt buộc trích dẫn nguồn → trả về `GenerationResult` (`answer`, `sources`, `retrieval_source=chunks[0]["retrieval_method"]`).

### 5.2. Hoàn thiện Chatbot Streamlit ([app.py](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/app.py))
- Gọi `generate_with_citation(query, top_k)`.
- Hiển thị câu trả lời Markdown.
- Hiển thị danh sách nguồn kèm `score`, `title`, `source`, `url` và `retrieval_method` trong `st.expander("📚 Nguồn trích dẫn")`.
- Lưu giữ nguyên trạng cấu trúc trong `st.session_state.messages` để khi chat các tin nhắn cũ vẫn xem lại được citation.
- Chạy thử Streamlit:
  ```powershell
  uv run streamlit run app.py
  ```

---

## Bước 6: Đánh Giá A/B & Hoàn Thiện Báo Cáo
*(Tương ứng tài liệu VLearn Bài 6, 7 & RESULT.md)*

### 6.1. Xây dựng Golden Dataset
Mở [group_project/evaluation/golden_dataset.json](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/group_project/evaluation/golden_dataset.json), nhập tối thiểu **15 câu hỏi - đáp** dựa trên đúng nội dung corpus đã thu thập:
```json
[
  {
    "question": "Câu hỏi cụ thể dựa trên tài liệu...",
    "expected_answer": "Câu trả lời chính xác trích xuất từ tài liệu...",
    "expected_context": "Đoạn văn bản gốc chứa thông tin trả lời..."
  },
  ...
]
```

### 6.2. Chạy Đánh Giá So Sánh A/B
So sánh 2 cấu hình giữ nguyên mọi yếu tố (LLM, Prompt, Dataset, Top_K) chỉ đổi cơ chế retrieval:
- **Config A (Dense-only)**: `retrieve(query, top_k=5, use_reranking=False)`
- **Config B (Hybrid + RRF)**: `retrieve(query, top_k=5, use_reranking=True)`

Đo đạc 4 chỉ số (bằng `ragas` hoặc script đánh giá dựa trên LLM-as-a-judge):
1. **Faithfulness**: Mức độ câu trả lời bám sát vào context được cung cấp.
2. **Answer Relevance**: Câu trả lời có giải quyết đúng trọng tâm câu hỏi không.
3. **Context Recall**: Retrieval có lấy được đầy đủ bằng chứng cần thiết trong `expected_context` không.
4. **Context Precision**: Các đoạn chunk lấy về có tập trung hay chứa nhiều đoạn nhiễu.

### 6.3. Hoàn thiện báo cáo [RESULT.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/group_project/evaluation/RESULT.md)
Điền đầy đủ các mục trong template (bắt buộc xóa hết tất cả chữ `TODO` để vượt qua test):
- **Run information**: Model, commit, threshold,...
- **Overall scores**: Bảng điểm Config A, Config B và Delta ($B - A$).
- **A/B comparison**: Phân tích cấu hình nào tốt hơn và đánh đổi (trade-off về latency/chi phí).
- **Worst performers**: Liệt kê 3 trường hợp điểm thấp nhất, phân loại lỗi thuộc tầng nào (*retrieval / generation / data*) và nguyên nhân gốc rễ (*root cause*).
- **Recommendations**: Đề xuất cải tiến cụ thể và cách kiểm chứng.

---

## Bước 7: Kiểm Thử Toàn Diện & Chuẩn Bị Nộp Bài

### 7.1. Chạy toàn bộ Test Suite
Trước khi nộp bài, chạy toàn bộ các bài kiểm tra tự động:
```powershell
# 1. Kiểm tra toàn bộ hợp đồng interface & logic invariant
uv run pytest tests/test_contracts.py -v

# 2. Kiểm tra các tiêu chuẩn nghiệm thu (corpus, standardized data, golden dataset, RESULT.md)
uv run pytest tests/test_acceptance.py -v

# 3. Kiểm tra toàn bộ repo
uv run pytest -q
```
*Tất cả các test phải pass 100%.*

### 7.2. Hoàn thiện báo cáo cá nhân
Mỗi thành viên copy [group_project/ịndividual/INDIVIDUAL_REPORT.md](file:///c:/Users/Vxtor/Documents/workspace/ai20k/K4-L3A-RAG-Pipeline/K4-L3A-RAG-Pipeline/group_project/%E1%BB%8Bndividual/INDIVIDUAL_REPORT.md) thành:
`reports/<student-id>-<short-name>.md` và điền chi tiết phần việc mình phụ trách, các commit tương ứng và các quyết định kỹ thuật.

### 7.3. Checklist an toàn trước khi nộp
- [ ] Không commit file `.env` hoặc để lộ bất kỳ API Key nào vào Git history.
- [ ] File `RESULT.md` không còn chữ `TODO`.
- [ ] Chatbot Streamlit chạy ổn định, demo được cả câu hỏi đúng chủ đề (có citation nguồn rõ ràng) và câu hỏi ngoài chủ đề (từ chối an toàn - safe refusal).

---

Bạn có thể bắt đầu với **Bước 1 (Setup môi trường với uv)** và **Bước 2 (Chọn chủ đề & thu thập dữ liệu)**. Khi bạn sẵn sàng cho bước nào hoặc cần hỗ trợ viết code chi tiết cho task nào, hãy cho tôi biết nhé!