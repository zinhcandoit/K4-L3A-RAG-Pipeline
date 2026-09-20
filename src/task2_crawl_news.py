"""
Task 2 — Crawl bài viết/thông báo.

Chủ đề nhóm: KINH DOANH GIA ĐÌNH (hộ kinh doanh, cá nhân kinh doanh).
Trục nội dung: bỏ thuế khoán từ 01/01/2026, chuyển sang kê khai, hóa đơn điện tử,
thủ tục đăng ký hộ kinh doanh.

Ưu tiên Crawl4AI nếu đã cài; nếu chưa có thì dùng đường dự phòng bằng
urllib + BeautifulSoup để Lab vẫn chạy được trước khi `uv sync`.

Cài browser trước khi chạy (đường Crawl4AI):
    python -m playwright install chromium

Chạy:
    python -m src.task2_crawl_news
"""

import asyncio
import json
import re
import time
import unicodedata
import urllib.request
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT = 45
MIN_CONTENT_CHARS = 200  # acceptance test yêu cầu bản chuẩn hóa >= 200 ký tự

ARTICLE_URLS = [
    "https://baochinhphu.vn/tu-khoan-sang-ke-khai-ho-kinh-doanh-hoi-co-quan-thue-tra-loi-102250612170705652.htm",
    "https://baochinhphu.vn/xoa-bo-thue-khoan-chuyen-sang-ke-khai-nganh-thue-cam-tay-chi-viec-ho-tro-ho-kinh-doanh-toi-da-102251030091555145.htm",
    "https://xaydungchinhsach.chinhphu.vn/huong-dan-ho-kinh-doanh-nop-thue-theo-phuong-phap-ke-khai-119250614094921101.htm",
    "https://xaydungchinhsach.chinhphu.vn/giai-dap-ve-ho-kinh-doanh-theo-phuong-phap-khoan-chuyen-sang-ke-khai-119250812140254931.htm",
    "https://xaydungchinhsach.chinhphu.vn/huong-dan-ho-kinh-doanh-su-dung-hoa-don-dien-tu-trong-ke-khai-thue-119260323165559066.htm",
    "https://vnexpress.net/bo-thue-khoan-ho-kinh-doanh-tinh-thue-the-nao-tu-2026-4992081.html",
    "https://vnexpress.net/ho-kinh-doanh-chuyen-sang-ke-khai-khong-can-mua-phan-mem-nop-le-phi-4958714.html",
    "https://tuoitre.vn/bo-thue-khoan-tu-2026-ho-kinh-doanh-can-chuan-bi-gi-ngay-tu-bay-gio-20251224151443742.htm",
    "https://tuoitre.vn/ho-kinh-doanh-chuyen-doi-so-bat-buoc-bo-thue-khoan-giup-lam-quen-may-tinh-tien-20250525091149628.htm",
    "https://tuoitre.vn/cuc-thue-go-vuong-cho-ho-kinh-doanh-20251103010447602.htm",
]

# Khối nội dung chính theo từng báo; thứ tự không quan trọng vì chọn khối nhiều chữ nhất.
CONTENT_SELECTORS = [
    "article",
    "[itemprop='articleBody']",
    ".detail-content",
    ".detail__content",
    ".detail-content-body",
    ".article-content",
    ".fck_detail",
    "#main-detail-body",
    ".detail-cmain",
    ".entry-content",
]

DROP_TAGS = ["script", "style", "noscript", "iframe", "svg", "form", "nav", "aside", "video"]
DROP_SELECTORS = [
    ".box-tinlienquan", ".tinlienquan", ".related-news", ".VCSortableInPreviewMode",
    ".social-share", ".banner", ".advertisement", ".ads", "figure .caption a",
]


def slugify(url: str) -> str:
    """Tên file ổn định theo URL, chạy lại không sinh bản trùng."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    tail = re.sub(r"\.(html?|aspx)$", "", tail, flags=re.I)
    tail = unicodedata.normalize("NFKD", tail).encode("ascii", "ignore").decode()
    tail = re.sub(r"[^a-zA-Z0-9]+", "-", tail).strip("-").lower()
    return tail[:80] or "article"


def _inline(node) -> str:
    from bs4 import NavigableString, Tag

    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in ("strong", "b"):
                parts.append(f"**{_inline(child).strip()}**")
            elif child.name in ("em", "i"):
                parts.append(f"*{_inline(child).strip()}*")
            elif child.name == "a":
                text = _inline(child).strip()
                href = child.get("href", "")
                parts.append(f"[{text}]({href})" if href and text else text)
            elif child.name == "br":
                parts.append("\n")
            else:
                parts.append(_inline(child))
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def _pick_container(soup):
    """Chọn khối có nhiều chữ trong thẻ <p> nhất."""
    best, best_score = None, 0
    for selector in CONTENT_SELECTORS:
        for node in soup.select(selector):
            score = sum(len(p.get_text(strip=True)) for p in node.find_all("p"))
            if score > best_score:
                best, best_score = node, score
    return best if best_score >= MIN_CONTENT_CHARS else soup.body


def parse_article(html: str, url: str) -> dict:
    """HTML -> {title, content_markdown, date_published}. Đây là phần parser của Task 2."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()

    def meta(*names: str) -> str:
        for name in names:
            node = soup.find("meta", attrs={"property": name}) or soup.find(
                "meta", attrs={"name": name}
            )
            if node and node.get("content", "").strip():
                return node["content"].strip()
        return ""

    title = meta("og:title", "twitter:title") or (
        soup.h1.get_text(" ", strip=True) if soup.h1 else ""
    )
    description = meta("og:description", "description")
    published = meta("article:published_time", "pubdate", "its_publish_date")

    container = _pick_container(soup)
    if container is None:
        raise ValueError("Không xác định được khối nội dung")

    for selector in DROP_SELECTORS:
        for node in container.select(selector):
            node.decompose()

    blocks = []
    if description:
        blocks.append(f"> {description}")

    for element in container.find_all(["h2", "h3", "h4", "p", "li", "figcaption", "blockquote"]):
        # bỏ phần tử lồng nhau đã được cha xử lý
        if element.find_parent(["li", "blockquote"]) is not None and element.name == "p":
            continue
        text = _inline(element)
        if not text or len(text) < 2:
            continue
        if element.name in ("h2", "h3", "h4"):
            blocks.append(f"{'#' * (int(element.name[1]) )} {text}")
        elif element.name == "li":
            blocks.append(f"- {text}")
        elif element.name == "figcaption":
            blocks.append(f"*{text}*")
        elif element.name == "blockquote":
            blocks.append(f"> {text}")
        else:
            blocks.append(text)

    # bỏ đoạn lặp liên tiếp do markup lồng nhau
    deduped = []
    for block in blocks:
        if not deduped or block != deduped[-1]:
            deduped.append(block)

    content = re.sub(r"\n{3,}", "\n\n", "\n\n".join(deduped)).strip()
    if len(content) < MIN_CONTENT_CHARS:
        raise ValueError(f"Nội dung quá ngắn ({len(content)} ký tự)")

    return {"title": title or url, "content_markdown": content, "date_published": published}


def _decompress(raw: bytes, encoding: str) -> bytes:
    """Nhiều báo trả gzip/deflate kể cả khi client không xin; phải giải nén trước khi parse."""
    encoding = (encoding or "").lower().strip()
    if encoding == "gzip":
        import gzip

        return gzip.decompress(raw)
    if encoding == "deflate":
        import zlib

        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    if encoding == "br":
        try:
            import brotli
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Response nén brotli, cần `pip install brotli`") from error
        return brotli.decompress(raw)
    return raw


def _looks_like_text(value: str) -> bool:
    """Chặn trường hợp lưu nhầm byte nén/nhị phân thành 'nội dung'."""
    if not value:
        return False
    sample = value[:4000]
    bad = sum(1 for ch in sample if ch == "�" or (ord(ch) < 32 and ch not in "\t\n\r"))
    return bad / len(sample) < 0.02


def fetch(url: str, retries: int = 3) -> str:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "gzip, deflate",
                    "Accept-Language": "vi,en;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                raw = _decompress(response.read(), response.headers.get("Content-Encoding"))
                charset = response.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
            if not _looks_like_text(html):
                raise ValueError("Response không phải text sau khi giải nén")
            return html
        except Exception as error:  # timeout, reset, incomplete read...
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Tải thất bại sau {retries} lần: {last_error}")


async def crawl_article(url: str) -> dict:
    """Crawl 1 URL -> dict đủ metadata bắt buộc của Lab."""
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            markdown = str(getattr(result, "markdown", "") or "").strip()
            if len(markdown) >= MIN_CONTENT_CHARS:
                metadata = getattr(result, "metadata", None) or {}
                return {
                    "url": url,
                    "title": metadata.get("title") or url,
                    "date_crawled": datetime.now().isoformat(timespec="seconds"),
                    "content_markdown": markdown,
                    "date_published": metadata.get("published_time", ""),
                    "extractor": "crawl4ai",
                }
    except ImportError:
        pass  # chưa cài crawl4ai -> dùng đường dự phòng

    parsed = await asyncio.to_thread(lambda: parse_article(fetch(url), url))
    return {
        "url": url,
        "title": parsed["title"],
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": parsed["content_markdown"],
        "date_published": parsed["date_published"],
        "extractor": "urllib+bs4",
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved, failed = 0, []
    for url in ARTICLE_URLS:
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"{slugify(url)}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  saved: {output.name}  ({len(article['content_markdown'])} ký tự)")
            saved += 1
        except Exception as error:
            print(f"  FAILED: {url} — {error}")
            failed.append(url)

    print(f"\nĐã lưu {saved}/{len(ARTICLE_URLS)} bài vào {DATA_DIR}")
    if failed:
        print("Cần xem lại:")
        for url in failed:
            print(f"  - {url}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
