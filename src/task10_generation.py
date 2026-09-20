"""
Task 10 — Generation có citation.

Luồng: retrieve -> reorder chống lost-in-the-middle -> format context có title
và source -> gọi provider trong .env -> trả answer, sources, retrieval_source.

Thiếu evidence hoặc provider lỗi thì trả safe refusal, tuyệt đối không bịa.
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
Mỗi khẳng định phải có citation ghi đúng tên file trong trường Source, dạng
[Nguồn: <tên file>]. Không đánh số tài liệu, không trích nguồn ngoài context.
Nếu thiếu evidence, hãy từ chối xác minh."""

REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label.

    Label dùng chính tên file làm khóa citation, không dùng số thứ tự: context đã
    bị reorder_for_llm() đảo thứ tự, nên "Document 2" sẽ không khớp với sources[1]
    mà UI hiển thị. Trích theo tên file thì đối chiếu được bất kể thứ tự.
    """
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def cited_sources(answer: str, sources: list[dict]) -> list[str]:
    """Các file thực sự được trích trong câu trả lời, để UI đánh dấu."""
    return [
        item["metadata"]["source"]
        for item in sources
        if item["metadata"]["source"] in answer
    ]


def _call_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    except Exception as error:
        # Một số model đời mới chỉ nhận temperature/top_p mặc định
        if "unsupported" not in str(error).lower() and "temperature" not in str(error).lower():
            raise
        response = client.chat.completions.create(model=LLM_MODEL, messages=messages)
    return (response.choices[0].message.content or "").strip()


def _call_gemini(system_prompt: str, user_message: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        ),
    )
    return (response.text or "").strip()


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    import anthropic

    response = anthropic.Anthropic().messages.create(
        model=LLM_MODEL,
        max_tokens=2048,
        system=system_prompt,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    if not LLM_MODEL:
        raise ValueError("Chưa đặt LLM_MODEL trong .env")

    dispatch = {
        "openai": _call_openai,
        "gemini": _call_gemini,
        "anthropic": _call_anthropic,
    }
    handler = dispatch.get(LLM_PROVIDER)
    if handler is None:
        raise ValueError(f"LLM_PROVIDER không hỗ trợ: {LLM_PROVIDER}")
    return handler(system_prompt, user_message)


def _retrieval_source(chunks: list[dict]) -> str:
    """Map retrieval_method của chunk sang contract RetrievalSource."""
    if not chunks:
        return "none"
    return "pageindex" if chunks[0]["retrieval_method"] == "pageindex" else "hybrid"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {"answer": REFUSAL, "sources": [], "retrieval_source": "none"}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:
        # Provider lỗi thì vẫn trả nguồn đã tìm được, nhưng không bịa câu trả lời
        print(f"[generation] Provider lỗi: {error}")
        return {
            "answer": REFUSAL,
            "sources": chunks,
            "retrieval_source": _retrieval_source(chunks),
        }

    if not answer:
        answer = REFUSAL

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": _retrieval_source(chunks),
    }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Từ 2026 hộ kinh doanh nộp thuế theo phương pháp nào?"
    result = generate_with_citation(question)
    print(f"Q: {question}\n")
    print(result["answer"])
    print(f"\n-- Nguồn ({result['retrieval_source']}) --")
    for item in result["sources"]:
        print(f"  {item['metadata']['source']}  (score {item['score']:.4f})")
