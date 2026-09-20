"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Chủ đề nhóm: KINH DOANH GIA ĐÌNH (hộ kinh doanh, cá nhân kinh doanh).

Nguồn: Hệ thống văn bản pháp quy của Chính phủ (vanban.chinhphu.vn).
Mỗi trang văn bản có đính kèm file PDF gốc trên datafiles.chinhphu.vn;
script tìm link đó rồi tải PDF về data/landing/legal/.

Lý do không dùng thuvienphapluat.vn: site trả 403 cho crawler.
Theo hướng dẫn Lab, gặp site chặn thì đổi nguồn công khai khác, không vượt WAF.

LƯU Ý: PDF ký số trên datafiles.chinhphu.vn là BẢN SCAN (không có text layer),
nên mỗi nguồn khai thêm "fulltext" — trang toàn văn HTML trên
xaydungchinhsach.chinhphu.vn. Task 3 lấy text từ đó; PDF giữ vai trò bản gốc
có chữ ký để đối chiếu và để thỏa acceptance test.

Chạy:
    python -m src.task1_collect_legal_docs
"""

import re
import sys
import urllib.request
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT = 60
MIN_PDF_BYTES = 1024  # acceptance test yêu cầu file > 1KB

# Văn bản nền của chủ đề. doc_no dùng để đối chiếu lại ở Task 3.
LEGAL_SOURCES = [
    {
        "slug": "nq-198-2025-qh15-kinh-te-tu-nhan",
        "doc_no": "198/2025/QH15",
        "title": "Nghị quyết 198/2025/QH15 về cơ chế, chính sách đặc biệt phát triển kinh tế tư nhân",
        "page": "https://vanban.chinhphu.vn/?pageid=27160&docid=213695",
        "fulltext": "https://xaydungchinhsach.chinhphu.vn/nghi-quyet-198-2025-qh15-ve-mot-so-co-che-chinh-sach-dac-biet-phat-trien-kinh-te-tu-nhan-119250517191622422.htm",
    },
    {
        "slug": "nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh",
        "doc_no": "168/2025/NĐ-CP",
        "title": "Nghị định 168/2025/NĐ-CP về đăng ký doanh nghiệp, đăng ký hộ kinh doanh",
        "page": "https://vanban.chinhphu.vn/?pageid=27160&docid=214334",
        "fulltext": "https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-168-2025-nd-cp-ve-dang-ky-doanh-nghiep-119250702175708554.htm",
    },
    {
        "slug": "nd-68-2026-chinh-sach-thue-ho-kinh-doanh",
        "doc_no": "68/2026/NĐ-CP",
        "title": "Nghị định 68/2026/NĐ-CP về chính sách thuế và quản lý thuế với hộ kinh doanh, cá nhân kinh doanh",
        "page": "https://vanban.chinhphu.vn/?pageid=27160&docid=217111",
        "fulltext": "https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-68-2026-nd-cp-quy-dinh-ve-chinh-sach-thue-quan-ly-thue-voi-ho-kinh-doanh-119260306102906789.htm",
    },
    {
        "slug": "tt-18-2026-btc-thu-tuc-thue-ho-kinh-doanh",
        "doc_no": "18/2026/TT-BTC",
        "title": "Thông tư 18/2026/TT-BTC về hồ sơ, thủ tục quản lý thuế với hộ kinh doanh, cá nhân kinh doanh",
        "page": "https://vanban.chinhphu.vn/?pageid=27160&docid=217174",
        "fulltext": "https://xaydungchinhsach.chinhphu.vn/thong-tu-18-2026-tt-btc-quy-dinh-ve-ho-so-thu-tuc-quan-ly-thue-voi-ho-ca-nhan-kinh-doanh-119260309174321852.htm",
    },
    {
        "slug": "tt-68-2025-btc-bieu-mau-dang-ky-ho-kinh-doanh",
        "doc_no": "68/2025/TT-BTC",
        "title": "Thông tư 68/2025/TT-BTC ban hành biểu mẫu đăng ký doanh nghiệp, đăng ký hộ kinh doanh",
        "page": "https://vanban.chinhphu.vn/?pageid=27160&docid=214411",
        "fulltext": "https://xaydungchinhsach.chinhphu.vn/toan-van-thong-tu-ban-hanh-bieu-mau-su-dung-trong-dang-ky-doanh-nghiep-dang-ky-ho-kinh-doanh-119260910084200046.htm",
    },
    {
        "slug": "nd-141-2026-sua-doi-nd-68-2026",
        "doc_no": "141/2026/NĐ-CP",
        "title": "Nghị định 141/2026/NĐ-CP sửa đổi Nghị định 68/2026/NĐ-CP về chính sách thuế hộ kinh doanh",
        "page": "https://vanban.chinhphu.vn/?pageid=27160&docid=217960",
    },
    {
        "slug": "nd-296-2026-sua-doi-nd-168-2025",
        "doc_no": "296/2026/NĐ-CP",
        "title": "Nghị định 296/2026/NĐ-CP sửa đổi Nghị định 168/2025/NĐ-CP về đăng ký doanh nghiệp",
        "page": "https://vanban.chinhphu.vn/?pageid=27160&docid=218986",
    },
]

ATTACHMENT_PATTERN = re.compile(
    r'https?://datafiles\.chinhphu\.vn/[^\s"\'<>]+?\.(?:pdf|doc|docx)', re.I
)


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def resolve_attachment(page_url: str) -> str:
    """Đọc trang văn bản và lấy link file gốc đính kèm."""
    html = fetch(page_url).decode("utf-8", errors="replace")
    match = ATTACHMENT_PATTERN.search(html)
    if not match:
        raise LookupError(f"Không tìm thấy file đính kèm trong {page_url}")
    return match.group(0)


def download_documents() -> list[dict]:
    """Tải file gốc của từng văn bản vào data/landing/legal/."""
    collected = []

    for source in LEGAL_SOURCES:
        attachment = resolve_attachment(source["page"])
        suffix = Path(attachment).suffix.lower() or ".pdf"
        target = DATA_DIR / f"{source['slug']}{suffix}"

        if target.exists() and target.stat().st_size > MIN_PDF_BYTES:
            print(f"  skip (đã có): {target.name}")
            collected.append({**source, "path": target})
            continue

        payload = fetch(attachment)
        if len(payload) <= MIN_PDF_BYTES:
            raise ValueError(f"File quá nhỏ, nghi ngờ lỗi tải: {attachment}")
        if suffix == ".pdf" and not payload.startswith(b"%PDF"):
            raise ValueError(f"Không phải PDF hợp lệ: {attachment}")

        target.write_bytes(payload)
        print(f"  saved: {target.name}  ({len(payload) / 1024:.0f} KB)  <- {attachment}")
        collected.append({**source, "path": target})

    return collected


def write_manifest(collected: list[dict]) -> None:
    """Ghi lại nguồn của từng file để Task 3 đối chiếu và để báo cáo truy vết."""
    import json

    manifest = DATA_DIR / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "file": item["path"].name,
                    "doc_no": item["doc_no"],
                    "title": item["title"],
                    "source_page": item["page"],
                    "fulltext_page": item.get("fulltext", ""),
                }
                for item in collected
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    setup_directory()
    try:
        collected = download_documents()
    except Exception as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        raise SystemExit(1)
    write_manifest(collected)
    print(f"\nTổng: {len(collected)} tài liệu trong {DATA_DIR}")
