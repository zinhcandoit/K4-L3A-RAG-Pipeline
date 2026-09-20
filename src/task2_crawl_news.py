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
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # Luật Thuế thu nhập doanh nghiệp
    "https://baochinhphu.vn/de-xuat-quy-dinh-moi-ve-uu-dai-thue-thu-nhap-doanh-nghiep-102260904162828502.htm",
    # Luật Giao dịch điện tử
    "https://baochinhphu.vn/ke-hoach-trien-khai-thi-hanh-luat-giao-dich-dien-tu-102231013205019682.htm",
    # Luật Doanh nghiệp
    "https://baochinhphu.vn/chu-tich-kiem-tong-giam-doc-co-duoc-lam-nguoi-dai-dien-theo-phap-luat-10226052207284733.htm",
    "https://baochinhphu.vn/tao-can-cu-phap-ly-de-trien-khai-cac-quy-dinh-cua-luat-giao-dich-dien-tu-102240216153251886.htm",
    "https://thuehaiquan.tapchikinhtetaichinh.vn/luat-thuong-mai-dien-tu-thiet-lap-chuan-muc-moi-cho-thi-truong-online-142525.html"
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

    # Fallback: requests + BeautifulSoup + markdownify
    import re

    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title = soup.title.get_text(strip=True) if soup.title else "Unknown"

    # Xoa cac tag khong can thiet
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                              "aside", "form", "iframe", "noscript", "svg"]):
        tag.decompose()

    # Tim phan noi dung bai viet (thu cac selector pho bien)
    article_body = (
        soup.find("div", class_=re.compile(r"detail[-_]?content|article[-_]?body|post[-_]?content", re.I))
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content[-_]?detail|entry[-_]?content|news[-_]?content", re.I))
        or soup.find("div", {"id": re.compile(r"content|article", re.I)})
    )

    if article_body:
        # Xoa cac phan tu menu/sidebar con sot lai trong article
        for junk in article_body.find_all(class_=re.compile(
            r"relate|sidebar|breadcrumb|share|social|comment|advert|banner|menu",
            re.I,
        )):
            junk.decompose()
        content = md(str(article_body), strip=["img"]).strip()
    else:
        # Fallback: lay body nhung bo cac phan rac
        body = soup.find("body")
        if body:
            for junk in body.find_all(class_=re.compile(
                r"header|footer|sidebar|menu|nav|breadcrumb|share|social|comment|advert|banner",
                re.I,
            )):
                junk.decompose()
            content = md(str(body), strip=["img"]).strip()
        else:
            content = md(str(soup), strip=["img"]).strip()

    # Clean: xoa CSS/JS artifacts con sot
    content = re.sub(r"@font-face\{[^}]+\}", "", content)
    content = re.sub(r"\.[a-zA-Z_][\w-]*\{[^}]+\}", "", content)
    content = re.sub(r":root\{[^}]+\}", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

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
