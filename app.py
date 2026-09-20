import os
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation, generate_with_jina

load_dotenv()

st.set_page_config(
    page_title="Hệ thống Hỏi Đáp Pháp Luật & Chính Sách (RAG)",
    page_icon="⚖️",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

jina_key = os.getenv("JINA_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Cấu hình RAG")
    st.caption("Pipeline RAG kết hợp Hybrid Retrieval & Reranker")

    rerank_mode = st.radio(
        "Mô hình Rerank:",
        ["RRF (Reciprocal Rank Fusion)", "Jina Reranker API (jina-reranker-v2-base-multilingual)"],
        index=0,
    )
    top_k = st.slider("Số lượng Chunks truy vấn (Top-K)", min_value=1, max_value=10, value=5)

    if "Jina" in rerank_mode and not jina_key:
        st.warning("⚠️ Chưa cấu hình JINA_API_KEY trong .env. Hệ thống sẽ tự động fallback sang thứ hạng gốc/RRF.")

    st.markdown("---")
    if st.button("🗑️ Xoá lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()

st.title("⚖️ Trợ Lý Pháp Luật & Chính Sách Doanh Nghiệp")
st.caption("Tra cứu và giải đáp dựa trên dữ liệu văn bản pháp luật và tin tức chính thống đã chuẩn hoá.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 Nguồn trích dẫn ({len(message['sources'])} chunks) — Phương pháp: {message.get('retrieval_source', 'N/A')}"):
                for idx, src in enumerate(message["sources"], 1):
                    meta = src.get("metadata", {})
                    st.markdown(
                        f"**{idx}. {meta.get('title', 'Tài liệu')}** (`{meta.get('source', '')}`)  \n"
                        f"- **Điểm số:** `{src.get('score', 0):.4f}` | **Phương pháp:** `{src.get('retrieval_method', '')}`  \n"
                        f"- **Nội dung:** {src.get('content', '')}"
                    )

query = st.chat_input("Nhập câu hỏi pháp luật hoặc chính sách...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu tài liệu và sinh câu trả lời..."):
            if "Jina" in rerank_mode:
                result = generate_with_jina(query, top_k=top_k)
            else:
                result = generate_with_citation(query, top_k=top_k)

            answer = result["answer"]
            sources = result.get("sources", [])
            retrieval_source = result.get("retrieval_source", "none")

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Nguồn trích dẫn ({len(sources)} chunks) — Phương pháp: {retrieval_source}"):
                    for idx, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        st.markdown(
                            f"**{idx}. {meta.get('title', 'Tài liệu')}** (`{meta.get('source', '')}`)  \n"
                            f"- **Điểm số:** `{src.get('score', 0):.4f}` | **Phương pháp:** `{src.get('retrieval_method', '')}`  \n"
                            f"- **Nội dung:** {src.get('content', '')}"
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
