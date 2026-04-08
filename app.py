import streamlit as st
import os
from rag_engine import (
    load_and_index_documents,
    load_existing_index,
    build_qa_chain,
    get_answer
)

# --- Page config ---
st.set_page_config(
    page_title="Knowledge Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Internal Knowledge Agent")
st.markdown("*Just prototyping and understanding how RAG works :)*")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Setup")
    
    if st.button("📥 Index Documents", type="primary"):
        with st.spinner("Indexing your documents... (2-5 min first time)"):
            vectorstore = load_and_index_documents()
            st.session_state.vectorstore = vectorstore
            st.session_state.chain = build_qa_chain(vectorstore)
        st.success("✅ Documents indexed!")

    if st.button("📂 Load Existing Index"):
        if os.path.exists("chroma_db"):
            with st.spinner("Loading index..."):
                vectorstore = load_existing_index()
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = build_qa_chain(vectorstore)
            st.success("✅ Index loaded!")
        else:
            st.error("No index found. Click 'Index Documents' first.")

    
# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if "chain" not in st.session_state:
        st.error("⚠️ Please index or load documents first (use the sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_answer(st.session_state.chain, prompt)

            st.markdown(answer)

            if sources:
                with st.expander("📎 Sources"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** `{src['file']}` — Page {src['page']}")
                        st.caption(src['snippet'])

            full_response = answer
            if sources:
                full_response += "\n\n*Sources cited above.*"
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
