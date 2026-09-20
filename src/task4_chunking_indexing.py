"""
Task 4 — Chunking, embedding và indexing.

Đọc Markdown trong data/standardized/, chunk bằng RecursiveCharacterTextSplitter,
embed bằng một provider duy nhất rồi upsert vào ChromaDB (cosine).

ID chunk ổn định theo đường dẫn file + chunk_index nên chạy lại pipeline chỉ
upsert đè, không sinh bản trùng. Task 5 import lại embed_texts()/get_collection()
từ module này để query dùng đúng không gian vector với index.

Chạy:
    python -m src.task4_chunking_indexing
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

EMBED_BATCH_SIZE = 32

# Header do Task 3 ghi: "**Nguồn:** <url>" (legal) hoặc "**Source:** <url>" (news).
URL_PATTERN = re.compile(r"^\*\*(?:Nguồn|Source):\*\*\s*(\S+)", re.M)
TITLE_PATTERN = re.compile(r"^#\s+(.+)$", re.M)

_model_cache: dict[str, object] = {}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text. Task 5 dùng chung hàm này để query cùng không gian vector."""
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = _model_cache.get(EMBEDDING_MODEL)
        if model is None:
            model = SentenceTransformer(EMBEDDING_MODEL)
            _model_cache[EMBEDDING_MODEL] = model
        return model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False).tolist()

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        client = genai.Client()
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [item.values for item in response.embeddings]

    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


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
    documents = []

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        doc_type = "legal" if "legal" in path.parts else "news"
        url_match = URL_PATTERN.search(content)
        title_match = TITLE_PATTERN.search(content)

        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": title_match.group(1).strip() if title_match else path.stem,
                    "doc_type": doc_type,
                    "url": url_match.group(1) if url_match else None,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            if not text.strip():
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _to_chroma_metadata(metadata: dict) -> dict:
    """Chroma không nhận None; giữ url rỗng rồi Task 5 đổi ngược lại thành None."""
    return {key: ("" if value is None else value) for key, value in metadata.items()}


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        raise ValueError("Không có chunk nào để index; kiểm tra data/standardized/")

    collection = get_collection()
    for start in range(0, len(chunks), 256):
        batch = chunks[start : start + 256]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_to_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Documents: {len(documents)}")

    chunks = chunk_documents(documents)
    print(f"Chunks: {len(chunks)} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks -> {CHROMA_DIR}")

    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk["metadata"]["doc_type"]] = by_type.get(chunk["metadata"]["doc_type"], 0) + 1
    print(f"Theo doc_type: {by_type}")


if __name__ == "__main__":
    run_pipeline()
