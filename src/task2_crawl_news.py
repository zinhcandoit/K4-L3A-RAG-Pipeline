"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
import sys
from pathlib import Path

# Windows console mặc định dùng cp1252, không hỗ trợ tiếng Việt
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # Luật Thuế thu nhập doanh nghiệp
    "https://baochinhphu.vn/de-xuat-quy-dinh-moi-ve-uu-dai-thue-thu-nhap-doanh-nghiep-102260904162828502.htm",
    # Luật Giao dịch điện tử
    "https://baochinhphu.vn/ke-hoach-trien-khai-thi-hanh-luat-giao-dich-dien-tu-102231013205019682.htm",
    # Luật Doanh nghiệp
    "https://baochinhphu.vn/chu-tich-kiem-tong-giam-doc-co-duoc-lam-nguoi-dai-dien-theo-phap-luat-10226052207284733.htm",
    # Luật Thuế giá trị gia tăng
    "https://baochinhphu.vn/giam-thue-gia-tri-gia-tang-tu-01-7-2025-den-het-31-12-2026-10225070118590677.htm",
    # Luật Chứng khoán
    "https://tapchitaichinh.vn/hoan-thien-hon-nua-khung-phap-ly-cho-nha-dau-tu-nuoc-ngoai-tren-thi-truong-chung-khoan.html",
]


async def crawl_article(url: str) -> dict:
    """Crawl một bài viết và trả về dict với url, title, date_crawled, content_markdown."""
    from datetime import datetime

    # Thử crawl4ai trước (headless browser, tốt cho JS-heavy sites)
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = "Unknown"
            if result.metadata and isinstance(result.metadata, dict):
                title = result.metadata.get("title", "Unknown")
            content = result.markdown or ""
            if len(content.strip()) > 100:
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": content,
                }
    except Exception as e:
        print(f"  crawl4ai failed for {url}: {e}, trying requests fallback...")

    # Fallback: requests + markdownify
    import re

    import requests
    from markdownify import markdownify as md

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; StudentBot/1.0)"},
    )
    response.raise_for_status()
    html = response.text

    # Extract title from HTML
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Unknown"

    content = md(html, strip=["script", "style", "nav", "footer", "header"])

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content.strip(),
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
