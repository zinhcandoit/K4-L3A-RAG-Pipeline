import os
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot - Chính sách & Thuế Hộ Kinh Doanh",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar cấu hình
with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Hệ thống giải đáp chính sách và thuế hộ kinh doanh")
    st.markdown("---")

    st.subheader("Cấu hình truy xuất")
    top_k = st.slider("Số lượng tài liệu (top_k)", min_value=1, max_value=10, value=4)

    st.markdown("---")
    st.subheader("Thông số hệ thống")
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").upper()
    llm_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    emb_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    st.caption(f"LLM: {llm_provider} ({llm_model})")
    st.caption(f"Embedding: {emb_model}")
    st.caption("Retrieval: Hybrid (Chroma Dense + BM25 Lexical + RRF)")
    st.caption("Fallback: PageIndex / Safe Refusal")

    st.markdown("---")
    if st.button("Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Tiêu đề chính
st.title("RAG Chatbot - Chính sách & Thuế Hộ Kinh Doanh")
st.caption("Trả lời câu hỏi từ tài liệu pháp lý và tin tức chính thống có trích dẫn nguồn.")

# Hiển thị toàn bộ lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        retrieval_source = message.get("retrieval_source", "none")

        if sources:
            st.caption(f"Phương pháp truy xuất: {retrieval_source.upper()} | Trích dẫn: {len(sources)} nguồn")
            with st.expander(f"Nguồn tham khảo ({len(sources)} tài liệu)"):
                for idx, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    title = meta.get("title", "Tài liệu")
                    source_name = meta.get("source", "Nguồn")
                    doc_type = meta.get("doc_type", "Chính sách")
                    score = src.get("score", 0.0)
                    method = src.get("retrieval_method", "hybrid")
                    content = src.get("content", "")
                    url = meta.get("url")
                    url_link = f" | [Liên kết gốc]({url})" if url else ""

                    st.markdown(f"**[Document {idx}] {title}**")
                    st.markdown(f"*File: `{source_name}` | Loại: `{doc_type}` | Score: `{score:.4f}` | Method: `{method}`{url_link}*")
                    st.markdown(f"> {content[:350]}...")
                    st.markdown("---")

# Ô nhập tin nhắn luôn luôn hiển thị
query = st.chat_input("Nhập câu hỏi về chính sách, thuế hoặc đăng ký hộ kinh doanh...")

if query:
    # 1. Lưu câu hỏi của user vào session_state
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Hiển thị câu hỏi của user
    with st.chat_message("user"):
        st.markdown(query)

    # 3. Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            result = generate_with_citation(query, top_k=top_k)
            answer = result["answer"]
            sources = result["sources"]
            retrieval_source = result["retrieval_source"]

        st.markdown(answer)

        if sources:
            st.caption(f"Phương pháp truy xuất: {retrieval_source.upper()} | Trích dẫn: {len(sources)} nguồn")
            with st.expander(f"Nguồn tham khảo ({len(sources)} tài liệu)"):
                for idx, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    title = meta.get("title", "Tài liệu")
                    source_name = meta.get("source", "Nguồn")
                    doc_type = meta.get("doc_type", "Chính sách")
                    score = src.get("score", 0.0)
                    method = src.get("retrieval_method", "hybrid")
                    content = src.get("content", "")
                    url = meta.get("url")
                    url_link = f" | [Liên kết gốc]({url})" if url else ""

                    st.markdown(f"**[Document {idx}] {title}**")
                    st.markdown(f"*File: `{source_name}` | Loại: `{doc_type}` | Score: `{score:.4f}` | Method: `{method}`{url_link}*")
                    st.markdown(f"> {content[:350]}...")
                    st.markdown("---")

    # 4. Lưu câu trả lời của assistant vào session_state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
