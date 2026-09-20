"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

from pathlib import Path


from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách văn bản theo provider được cấu hình."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

    if provider == "gemini":
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env")
        client = genai.Client(api_key=api_key)
        
        # Batch requests with smaller batch size (20) to fit free tier rate limiter bucket
        all_embeddings: list[list[float]] = []
        batch_size = 20
        import time
        import re

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            max_retries = 20
            for attempt in range(max_retries):
                try:
                    res = client.models.embed_content(
                        model=model_name,
                        contents=batch,
                    )
                    for emb in res.embeddings:
                        all_embeddings.append(emb.values)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries - 1:
                        # Extract retryDelay if present
                        wait_sec = 50
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                        if match:
                            wait_sec = int(float(match.group(1))) + 3
                        print(f"Rate limited (429) on batch {i // batch_size + 1}, waiting {wait_sec}s for quota window...")
                        time.sleep(wait_sec)
                    else:
                        raise
            time.sleep(2.0)
        return all_embeddings

    elif provider in {"sentence_transformers", "local"}:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            return model.encode(texts).tolist()
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. Install it or set EMBEDDING_PROVIDER=gemini"
            )

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.embeddings.create(input=texts, model=model_name or "text-embedding-3-small")
        return [item.embedding for item in resp.data]

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        doc_type = "legal" if "legal" in path.parts else "news"
        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix()

        # Trích xuất title và url nếu có
        title = path.stem
        url = None

        lines = content.splitlines()
        for line in lines[:10]:
            trimmed = line.strip()
            if trimmed.startswith("# "):
                extracted_title = trimmed[2:].strip()
                if extracted_title:
                    title = extracted_title
            elif trimmed.startswith("**Source:**"):
                extracted_url = trimmed[len("**Source:**"):].strip()
                if extracted_url:
                    url = extracted_url

        documents.append({
            "id": doc_id,
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        doc_id = document["id"]
        meta = dict(document["metadata"])
        split_texts = splitter.split_text(document["content"])

        chunk_idx = 0
        for text in split_texts:
            clean_text = text.strip()
            if not clean_text:
                continue
            chunk_metadata = dict(meta)
            chunk_metadata["chunk_index"] = chunk_idx
            chunks.append({
                "id": f"{doc_id}::chunk-{chunk_idx}",
                "content": clean_text,
                "metadata": chunk_metadata,
            })
            chunk_idx += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk có caching."""
    if not chunks:
        return []

    import json
    import hashlib

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CHROMA_DIR / "embeddings_cache.json"
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    missing: list[tuple[dict, str]] = []
    for chunk in chunks:
        h = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
        if h in cache:
            chunk["embedding"] = cache[h]
        else:
            missing.append((chunk, h))

    if missing:
        cached_count = len(chunks) - len(missing)
        print(f"Embedding {len(missing)} chunks ({cached_count} loaded from cache)...")
        batch_size = 20
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            batch_texts = [item[0]["content"] for item in batch]
            vectors = embed_texts(batch_texts)
            for (chunk, h), vector in zip(batch, vectors):
                chunk["embedding"] = vector
                cache[h] = vector
            # Lưu cache từng batch
            try:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            except Exception:
                pass
            print(f"Embedded {min(i + batch_size, len(missing))}/{len(missing)} chunks")
    else:
        print(f"All {len(chunks)} chunks loaded from cache!")

    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    collection = get_collection()
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        # Chroma requires metadata values to be str, int, float or bool (not None)
        sanitized_metadatas = [
            {k: ("" if v is None else v) for k, v in chunk["metadata"].items()}
            for chunk in batch
        ]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=sanitized_metadatas,
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print(f"Generated embeddings for {len(embedded_chunks)} chunks")
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed {len(embedded_chunks)} chunks to ChromaDB")


if __name__ == "__main__":
    run_pipeline()
