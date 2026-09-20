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

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "")
        source = metadata.get("source", "")
        content = chunk.get("content", "")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL)

    if provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        client = genai.Client(api_key=api_key)
        model_name = model or "gemini-3.5-flash-lite"
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=config,
        )
        return response.text or ""

    elif provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
        model_name = model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""

    elif provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(api_key=api_key)
        model_name = model or "claude-3-5-sonnet-20241022"
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
        )
        return response.content[0].text or ""

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    if not query.strip():
        return {
            "answer": "Vui lòng nhập câu hỏi.",
            "sources": [],
            "retrieval_source": "none",
        }

    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Hãy trả lời câu hỏi trên dựa CHỈ vào context. Ghi rõ trích dẫn [Document X] sau mỗi thông tin."
    )

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        if not answer or not answer.strip():
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    except Exception as e:
        print(f"LLM call failed: {e}")
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source = "pageindex" if method == "pageindex" else "hybrid"

    return {
        "answer": answer.strip(),
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    result = generate_with_citation("Hộ kinh doanh nộp thuế theo phương pháp nào?")
    print("Answer:\n", result["answer"])
    print("\nSources count:", len(result["sources"]))
    print("Retrieval source:", result["retrieval_source"])
