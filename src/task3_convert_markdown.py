"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Quy định định dạng landing:
    - data/landing/legal/: chỉ chứa các file tài liệu gốc (.doc, .docx, .pdf), KHÔNG chứa file .md.
    - data/landing/news/: chỉ chứa kết quả crawl dạng JSON (.json), KHÔNG chứa file .md.
    - Tầng Markdown (.md) chỉ được tạo ra và lưu trữ tại data/standardized/.
"""

import re
import struct
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _clean_text(text: str) -> str:
    """Chuẩn hóa văn bản: dọn khoảng trắng thừa, giữ khoảng cách đoạn hợp lý."""
    # Chuẩn hóa ngắt dòng
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Xóa ký tự điều khiển lạ nhưng giữ lại \n, \t
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    # Xóa khoảng trắng/tab ở cuối mỗi dòng
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    # Gộp 3+ dòng trống liên tiếp thành 2 dòng trống (\n\n)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def _extract_docx(path: Path) -> str:
    """Convert .docx sang Markdown bằng MarkItDown, fallback qua python-docx."""
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(path))
        if result and result.text_content:
            return result.text_content
    except Exception:
        pass

    # Fallback to python-docx
    try:
        import docx
        doc = docx.Document(str(path))
        parts = []
        for p in doc.paragraphs:
            if p.text:
                parts.append(p.text)
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    parts.append(row_text)
        return "\n\n".join(parts)
    except Exception as e:
        raise RuntimeError(f"Không thể đọc file docx {path}: {e}")


def _extract_doc(path: Path) -> str:
    """Trích xuất văn bản từ file Word 97-2004 (.doc) định dạng OLE CFBF."""
    # Phương pháp 1: Đọc qua olefile và Piece Table theo chuẩn MS-DOC FIB
    try:
        import olefile
        if olefile.isOleFile(str(path)):
            ole = olefile.OleFileIO(str(path))
            word_doc = ole.openstream("WordDocument").read()
            flags = struct.unpack("<H", word_doc[0x000A:0x000C])[0]
            table_name = "1Table" if (flags & 0x0200) else "0Table"
            table_stream = ole.openstream(table_name).read()

            fcClx = struct.unpack("<I", word_doc[0x01A2:0x01A6])[0]
            lcbClx = struct.unpack("<I", word_doc[0x01A6:0x01AA])[0]
            clx = table_stream[fcClx:fcClx + lcbClx]
            ccpText = struct.unpack("<I", word_doc[0x004C:0x0050])[0]

            pos = 0
            full_text = []
            while pos < len(clx):
                clxt = clx[pos]
                pos += 1
                if clxt == 1:
                    pos += 2 + struct.unpack("<H", clx[pos:pos + 2])[0]
                elif clxt == 2:
                    lcb = struct.unpack("<I", clx[pos:pos + 4])[0]
                    pos += 4
                    plcfpcd = clx[pos:pos + lcb]
                    n = (lcb - 4) // 12
                    cps = [struct.unpack("<I", plcfpcd[i * 4:(i + 1) * 4])[0] for i in range(n + 1)]
                    pcd_start = 4 * (n + 1)
                    for i in range(n):
                        pcd = plcfpcd[pcd_start + i * 8:pcd_start + (i + 1) * 8]
                        fc = struct.unpack("<I", pcd[2:6])[0]
                        length = cps[i + 1] - cps[i]
                        if (fc & 0x40000000) != 0:
                            real_fc = (fc & ~0x40000000) // 2
                            raw = word_doc[real_fc:real_fc + length]
                            full_text.append(raw.decode("cp1252", errors="replace"))
                        else:
                            raw = word_doc[fc:fc + length * 2]
                            full_text.append(raw.decode("utf-16le", errors="replace"))
                    break

            text = "".join(full_text)[:ccpText]
            text = text.replace("\r", "\n").replace("\x0b", "\n").replace("\x07", "\t")
            if text.strip():
                return text
    except Exception:
        pass

    # Phương pháp 2: Fallback pywin32 / Word COM automation nếu có
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(str(path.resolve()))
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        if text.strip():
            return text
    except Exception:
        pass

    raise RuntimeError(f"Không thể đọc file doc {path}")


def _extract_pdf(path: Path) -> str:
    """Convert .pdf sang Markdown bằng MarkItDown."""
    from markitdown import MarkItDown
    md = MarkItDown()
    res = md.convert(str(path))
    return res.text_content


def convert_legal_docs() -> None:
    """Đọc .doc, .docx, .pdf từ landing/legal, convert sang Markdown, clean và ghi vào standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = {".doc", ".docx", ".pdf"}
    for path in sorted(legal_dir.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in allowed_exts:
            continue

        ext = path.suffix.lower()
        if ext == ".docx":
            raw_text = _extract_docx(path)
        elif ext == ".doc":
            raw_text = _extract_doc(path)
        elif ext == ".pdf":
            raw_text = _extract_pdf(path)
        else:
            continue

        cleaned = _clean_text(raw_text)
        dest = output_dir / f"{path.stem}.md"
        dest.write_text(cleaned, encoding="utf-8")
        saved = len(raw_text) - len(cleaned)
        tag = f" (-{saved:,}b)" if saved > 0 else ""
        print(f"  [OK] legal: {dest.name}{tag}")


def convert_news_articles() -> None:
    """Convert JSON crawl results (.json) thành Markdown với metadata header, đã clean."""
    import json

    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        header = (
            f"# {data['title']}\n"
            f"**Source:** {data['url']}\n"
            f"**Crawled:** {data['date_crawled']}\n---\n"
        )
        raw = header + data["content_markdown"]
        cleaned = _clean_text(raw)
        (output_dir / f"{path.stem}.md").write_text(cleaned, encoding="utf-8")
        print(f"  [OK] news: {path.stem}.md")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing sang standardized Markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
