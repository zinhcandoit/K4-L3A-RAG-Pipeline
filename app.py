"""Chatbot RAG — hỏi đáp về hộ kinh doanh (kinh doanh gia đình).

Chạy:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from src.task9_retrieval_pipeline import SCORE_THRESHOLD
from src.task10_generation import cited_sources, generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="Hỏi đáp hộ kinh doanh",
    page_icon="🏪",
    layout="wide",
)

BADGE = {
    "hybrid": ("🔀", "Hybrid (dense + BM25 + RRF)"),
    "pageindex": ("📄", "PageIndex fallback"),
    "none": ("⚠️", "Không tìm thấy bằng chứng"),
}


def render_sources(sources: list[dict], answer: str, retrieval_source: str) -> None:
    """Hiện nguồn, đánh dấu file thực sự được trích trong câu trả lời."""
    icon, label = BADGE.get(retrieval_source, ("", retrieval_source))
    st.caption(f"{icon} {label} · {len(sources)} đoạn ngữ cảnh")

    if not sources:
        return

    cited = set(cited_sources(answer, sources))
    for index, item in enumerate(sources, 1):
        metadata = item["metadata"]
        source = metadata["source"]
        mark = "✅ đã trích" if source in cited else "· không trích"
        header = f"{index}. {source} — score {item['score']:.4f} ({mark})"

        with st.expander(header):
            st.caption(
                f"**{metadata['title']}**  \n"
                f"Loại: `{metadata['doc_type']}` · chunk #{metadata['chunk_index']}"
            )
            if metadata.get("url"):
                st.caption(f"Nguồn gốc: {metadata['url']}")
            st.text(item["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🏪 Hỏi đáp hộ kinh doanh")
    st.caption(
        "Trả lời dựa trên văn bản pháp luật và bài báo về hộ kinh doanh, "
        "cá nhân kinh doanh — trọng tâm là bỏ thuế khoán từ 01/01/2026."
    )

    top_k = st.slider("Số chunks", 3, 10, 5)

    st.divider()
    st.caption("**Cấu hình**")
    st.caption(
        f"LLM: `{os.getenv('LLM_MODEL', '?')}`  \n"
        f"Embedding: `{os.getenv('EMBEDDING_MODEL', '?')}`  \n"
        f"Ngưỡng fallback: `{SCORE_THRESHOLD}`"
    )

    st.divider()
    st.caption("**Thử hỏi**")
    st.caption(
        "- Từ 2026 hộ kinh doanh nộp thuế thế nào?\n"
        "- Hồ sơ đăng ký hộ kinh doanh gồm những gì?\n"
        "- Khi nào phải dùng hóa đơn điện tử từ máy tính tiền?"
    )

    if st.button("Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Hỏi đáp hộ kinh doanh")
st.caption(
    "Chatbot chỉ trả lời trong phạm vi tài liệu đã thu thập. "
    "Câu ngoài phạm vi sẽ bị từ chối thay vì bịa."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message["content"],
                message.get("retrieval_source", "none"),
            )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm trong tài liệu..."):
            try:
                result = generate_with_citation(query, top_k=top_k)
            except Exception as error:
                # Pipeline lỗi thì báo rõ, không để giao diện crash
                result = {
                    "answer": f"Pipeline lỗi: {error}",
                    "sources": [],
                    "retrieval_source": "none",
                }

        st.markdown(result["answer"])
        render_sources(result["sources"], result["answer"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
