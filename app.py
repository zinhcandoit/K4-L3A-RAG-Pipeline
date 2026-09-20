import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Hỏi đáp trên tài liệu pháp lý và tin tức đã thu thập")
    top_k = st.slider("Số chunks", 3, 10, 5)
    if st.button("Xóa hội thoại"):
        st.session_state.messages = []
        st.rerun()


def render_sources(sources: list[dict], retrieval_source: str | None) -> None:
    """Hiển thị nguồn trích dẫn kèm điểm truy xuất."""
    if not sources:
        return
    with st.expander(f"Nguồn ({len(sources)}) — truy xuất: {retrieval_source}"):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            st.markdown(
                f"**[{index}] {metadata['title']}** · `{metadata['source']}` · "
                f"{source['retrieval_method']} · score {source['score']:.4f}"
            )
            if metadata.get("url"):
                st.markdown(metadata["url"])
            st.caption(source["content"][:500])


st.title("RAG Chatbot")
st.caption("Nhập câu hỏi; câu trả lời có trích dẫn nguồn từ tài liệu.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []), message.get("retrieval_source"))

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            from src.task10_generation import generate_with_citation

            with st.spinner("Đang truy xuất và tạo câu trả lời..."):
                result = generate_with_citation(query, top_k=top_k)
            answer = result["answer"]
            sources = result["sources"]
            retrieval_source = result["retrieval_source"]
        except Exception as error:
            answer = f"Đã xảy ra lỗi khi xử lý câu hỏi: {error}"
            sources, retrieval_source = [], "none"
        st.markdown(answer)
        render_sources(sources, retrieval_source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
