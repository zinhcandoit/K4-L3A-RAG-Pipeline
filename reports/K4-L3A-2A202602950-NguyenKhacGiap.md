# Individual contribution report

## Thông tin

- Họ và tên: Nguyễn Khắc Giáp
- Mã học viên: `<điền MSSV>`
- Nhóm: `<điền tên nhóm>` — nhóm làm chung toàn bộ pipeline và review chéo cho nhau; phần tôi nhận trách nhiệm chính là **Data** và **Retrieval**
- Repository/branch: https://github.com/zinhcandoit/K4-L3A-RAG-Pipeline — nhánh `feat/data-ho-kinh-doanh`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Chốt chủ đề + corpus | Chọn chủ đề hộ kinh doanh, trục "bỏ thuế khoán từ 01/01/2026"; loại 7 file markdown cũ có tên file không khớp nội dung | `44a15af` | Done |
| Task 1 — thu thập văn bản | Script dò link đính kèm trên `datafiles.chinhphu.vn`, tải 7 PDF ký số, kiểm magic `%PDF`, ghi `manifest.json` truy vết | `src/task1_collect_legal_docs.py` · `44a15af`, `cc254c0` | Done |
| Task 2 — crawler + parser | Crawl4AI + đường dự phòng `urllib`/bs4; parser chọn khối `<p>` nhiều chữ nhất, lọc nav/ads; thu 10 bài từ 4 báo | `src/task2_crawl_news.py` · `44a15af` | Done |
| Task 3 — chuẩn hóa | Convert PDF/JSON sang Markdown, idempotent theo mtime; đối chiếu số hiệu văn bản với nội dung trích ra | `src/task3_convert_markdown.py` · `44a15af` | Done |
| Task 4-6 — index + 2 đường tìm kiếm | Chunking 1163 chunk, embed, upsert Chroma cosine; dense search; BM25 | `src/task4..6` · `a218f86`, `0ed3a58` | Done |
| Task 7, 9 — RRF + pipeline | RRF không mutate list đầu vào; pipeline fuse đúng 1 lần, fallback đọc cosine score gốc | `src/task7`, `src/task9` · `a218f86` | Done |
| Task 8 — PageIndex | Upload + cache doc_id + parser node phòng thủ nhiều shape response | `src/task8_pageindex_vectorless.py` · `a218f86` | Partial — nhóm chưa có API key nên nhánh gọi thật chưa chạy được |
| Bước 6 — evaluation | Chưa làm | — | Blocked |

Ghi chú trung thực: các commit `44a15af`, `cc254c0`, `bff0681` có trailer `Co-Authored-By: Claude Opus 5` — phần code trong đó được viết với AI hỗ trợ dưới sự điều hướng và review của tôi. Các quyết định kỹ thuật, việc chẩn đoán lỗi và số liệu hiệu chỉnh bên dưới là phần tôi chịu trách nhiệm giải thích và chạy lại khi demo.

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Không OCR PDF luật; lấy text từ trang toàn văn HTML trên `xaydungchinhsach.chinhphu.vn`, giữ PDF ký số làm bản gốc.
   **Lý do/evidence:** MarkItDown và pypdf đều trả **0 ký tự** trên cả 7 PDF. Kiểm tra cấu trúc cho thấy không có `/Font`, chỉ có `/XObject` kiểu `DCTDecode` và `CCITTFaxDecode` — PDF ký số của Chính phủ là ảnh scan.
   **Trade-off:** File `.md` không sinh ra từ chính byte của `.pdf` nằm cạnh nó. Đổi lại tránh được sai số dấu thanh của OCR trên văn bản thuế — sai một con số là citation dẫn sai. Đã ghi cả hai URL vào header mỗi file và vào `manifest.json` để vẫn truy vết được.

2. **Quyết định:** Đổi embedding từ `BAAI/bge-m3` sang `text-embedding-3-small` của OpenAI.
   **Lý do/evidence:** Đo trực tiếp bằng `curl` tới CDN HuggingFace: **80.880 B/s**. Model 2,27 GB tức ~8 tiếng tải, không khả thi trong buổi lab. Tắt xet (`HF_HUB_DISABLE_XET=1`) không cải thiện vì nghẽn ở đường mạng.
   **Trade-off:** Mất tính chạy offline và phụ thuộc API bên ngoài; bù lại index xong trong ~1 phút, chi phí đo bằng `tiktoken` là 188.123 token ≈ **$0,0038**. Số chiều đổi 1024 → 1536, đã cập nhật `EMBEDDING_DIM`.

## Kiểm thử và kết quả

- **Test:** `pytest -q` → **18 passed, 2 failed**. Hai test fail là `test_golden_dataset_has_15_grounded_cases` và `test_evaluation_report_is_completed`, thuộc bước 6 chưa làm. Toàn bộ 15 contract test và 3 acceptance test về dữ liệu đều pass.
- **Hiệu chỉnh `SCORE_THRESHOLD`** bằng query in-domain và out-of-domain:

  | Query | Best dense score |
  |---|---|
  | "hộ kinh doanh nộp thuế 2026" (in-domain) | 0,7930 |
  | "chăm sóc lan hồ điệp" (ngoài miền) | 0,3804 |
  | "Messi World Cup 2022" (ngoài miền) | 0,2193 |

  Chọn **0,45**. Mức mặc định 0,3 không chặn được query ngoài miền thứ nhất. Không khẳng định ngưỡng này đúng cho corpus khác.
- **Kết quả trước/sau:** trước khi đổi chủ đề, `landing/legal` có 7 file `.md` mà 6/7 có tên khác hẳn nội dung bên trong (ví dụ `law-2024-luat-thue-gia-tri-gia-tang.md` chứa Luật Di sản văn hóa); acceptance test về legal fail vì đếm `.pdf/.doc/.docx` ra 0. Sau khi làm lại: 5 văn bản đúng chủ đề, test pass.

**Ba lỗi tự phát hiện và cách xử lý:**

1. *Crawler lưu byte nhị phân.* Bốn file JSON đầu chứa rác vì server trả gzip mà `urllib` không giải nén. Thêm `_decompress()` theo header `Content-Encoding` và guard `_looks_like_text()` chặn ghi nội dung phi văn bản, kèm retry cho lỗi mạng.
2. *BM25 trả rỗng trên corpus nhỏ.* `BM25Okapi` cho IDF = 0 khi một term xuất hiện ở 1/2 document (`log(1,5) − log(1,5)`), nên filter `score <= 0` loại sạch kết quả. Sửa thành: chỉ lọc score dương khi thực sự có, không thì giữ thứ hạng thô.
3. *Citation trỏ sai nguồn.* `reorder_for_llm()` đảo context thành `[0,2,4,3,1]` nhưng `sources` trả theo score giảm dần, nên `[Document 2]` model sinh ra không khớp nguồn thứ 2 trên UI. Không thể trả `sources` theo thứ tự đã reorder vì contract bắt score giảm dần, nên đổi sang bắt model trích theo tên file `[Nguồn: <file>]` — đối chiếu được bất kể thứ tự.

## Điều còn hạn chế

- **Corpus mất cân đối nghiêm trọng:** NĐ 168/2025 chiếm **76,4%** tổng chunk (889/1163), trong khi NĐ 68/2026 — nghị định thuế, đúng trọng tâm chủ đề — chỉ có 9 chunk (0,8%) vì trang "toàn văn" của nó thực chất là bản tóm tắt 2.587 ký tự. Khi hỏi về hồ sơ đăng ký, cả 5 nguồn trả về đều từ một văn bản duy nhất.
- **Nếu có thêm thời gian, việc đầu tiên tôi làm:** cắt NĐ 168/2025 chỉ giữ Chương VIII (Hộ kinh doanh và đăng ký hộ kinh doanh). Đo được rằng 709/820 lần nhắc "hộ kinh doanh" nằm ở 30% cuối file, còn khoảng 30-70% file không nhắc chữ nào — tức vài trăm chunk thuần nội dung doanh nghiệp, là nhiễu với chủ đề. Sau đó chạy lại A/B để xem cân bằng corpus ảnh hưởng thế nào tới recall.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Nguyễn Khắc Giáp
