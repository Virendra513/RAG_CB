import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lg_backend_resume import (
    cb,
    get_first_human_message_10_words,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ─── Utility helpers ─────────────────────────────────────────────────────────────

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    _add_thread(thread_id)
    st.session_state["message_history"] = []


def _add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    state = cb.get_state(config={"configurable": {"thread_id": thread_id}})
    if not state or not hasattr(state, "values"):
        return []
    return state.values.get("messages", [])

# ─── Session initialisation ───────────────────────────────────────────────────────

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

_add_thread(st.session_state["thread_id"])

thread_key: str = st.session_state["thread_id"]
thread_docs: dict = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────────

st.sidebar.title("ESSMORATH-AI Assistant")
#st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Document status
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

# PDF uploader
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

# Past conversations
st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        label = get_first_human_message_10_words(tid) or str(tid)
        if st.sidebar.button(label, key=f"side-thread-{tid}"):
            selected_thread = tid

# ─── Main chat area ───────────────────────────────────────────────────────────────

st.title("Intelligent Multi Utility Chatbot")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder: dict = {"box": None}

        def ai_stream():
            for message_chunk, _ in cb.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Show a status box when a tool message arrives
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f" Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f" Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream only AI text tokens
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_reply = st.write_stream(ai_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label=" Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_reply}
    )

    # Show document metadata below the reply if a doc is indexed
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 Document: **{doc_meta.get('filename')}** — "
            f"{doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages"
        )

st.divider()

# ─── Thread switching ─────────────────────────────────────────────────────────────

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            temp_messages.append({"role": "assistant", "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()