import streamlit as st
from rag_pipeline import get_rag_chain

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("Chatbot: Ask about Romanian 2025 Elections")

qa_chain = get_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask something about the documents:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        answer = result["result"]

        # Save to chat history
        st.session_state.chat_history.append({"user": query, "bot": answer})

# Display history (latest at the bottom)
for chat in st.session_state.chat_history:
    st.markdown(f"""
    <div style='background-color:#f0f0f0; padding: 10px 15px; border-radius: 10px; margin-bottom: 10px; text-align: right'>
        <b>You:</b><br>{chat['user']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background-color:#e8f0fe; padding: 10px 15px; border-radius: 10px; margin-bottom: 10px; text-align: left'>
        <b>Bot:</b><br>{chat['bot']}
    </div>
    """, unsafe_allow_html=True)