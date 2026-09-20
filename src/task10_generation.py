"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve, retrieve_with_bge

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp văn bản pháp luật và tin tức chính thống.
Chỉ trả lời dựa trên context được cung cấp dưới đây.
Mỗi khẳng định hoặc thông tin trả lời phải trích dẫn nguồn (Title hoặc Source) tương ứng.
Nếu context không chứa đủ thông tin để trả lời câu hỏi, hãy trung thực thông báo: "Tôi không thể xác minh thông tin này từ nguồn hiện có." và tuyệt đối không bịa đặt thông tin."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context để giảm lost-in-the-middle."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label rõ ràng cho LLM trích dẫn."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", f"Doc-{index}")
        source = metadata.get("source", "unknown")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI (hoặc NVIDIA NIM), Gemini hoặc Anthropic theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    model_name = os.getenv("LLM_MODEL", LLM_MODEL)

    try:
        if provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not base_url and api_key.startswith("nvapi-"):
                base_url = "https://integrate.api.nvidia.com/v1"

            client = OpenAI(api_key=api_key, base_url=base_url)
            model = model_name or "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return response.choices[0].message.content or ""

        elif provider == "gemini":
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            client = genai.Client(api_key=api_key)
            model = model_name or "gemini-2.5-flash"
            response = client.models.generate_content(
                model=model,
                contents=f"{system_prompt}\n\n{user_message}",
            )
            return response.text or ""

        elif provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            client = anthropic.Anthropic(api_key=api_key)
            model = model_name or "claude-3-5-sonnet-20241022"
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=TEMPERATURE,
            )
            return response.content[0].text

        else:
            return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    except Exception as e:
        print(f"Error calling LLM provider '{provider}': {e}")
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult gồm answer, sources và retrieval_source."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Dưới đây là context tham khảo:\n\n{context}\n\nCâu hỏi: {query}"
    answer = call_llm(SYSTEM_PROMPT, user_message)

    if not answer or not answer.strip():
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0]["retrieval_method"],
    }


def generate_with_bge(query: str, top_k: int = TOP_K) -> dict:
    """Generation sử dụng BAAI/bge-reranker-v2-m3 reranker."""
    chunks = retrieve_with_bge(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Dưới đây là context tham khảo:\n\n{context}\n\nCâu hỏi: {query}"
    answer = call_llm(SYSTEM_PROMPT, user_message)

    if not answer or not answer.strip():
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0]["retrieval_method"],
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
