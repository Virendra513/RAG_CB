from __future__ import annotations

import json
import os
import random
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ─── HF Client ──────────────────────────────────────────────────────────────────

client1 = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

client2 =  InferenceClient(
    api_key=os.environ['HF_TOKEN'],
)
EMBEDDING_MODEL = "ibm-granite/granite-embedding-278m-multilingual"

# ─── Sentence-similarity retriever ──────────────────────────────────────────────
# Stores plain text chunks per thread; ranks them with sentence_similarity at
# query time — no FAISS / no feature_extraction needed.

class SimilarityRetriever:
    """Lightweight retriever backed by HF sentence_similarity."""

    def __init__(self, chunks: List[str], top_k: int = 4):
        self.chunks = chunks
        self.top_k  = top_k

    def invoke(self, query: str) -> List[str]:
        if not self.chunks:
            return []

        scores: List[float] = client1.sentence_similarity(
            {
                "source_sentence": query,
                "sentences": self.chunks,
            },
            model=EMBEDDING_MODEL,
        )

        ranked = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)
        return [text for _, text in ranked[: self.top_k]]


# ─── PDF Store ──────────────────────────────────────────────────────────────────

_THREAD_RETRIEVERS: Dict[str, SimilarityRetriever] = {}
_THREAD_METADATA:   Dict[str, dict]                 = {}


def _get_retriever(thread_id: Optional[str]) -> Optional[SimilarityRetriever]:
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        docs        = PyPDFLoader(temp_path).load()
        chunks      = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        ).split_documents(docs)
        chunk_texts = [c.page_content for c in chunks]

        _THREAD_RETRIEVERS[str(thread_id)] = SimilarityRetriever(chunk_texts, top_k=4)
        _THREAD_METADATA[str(thread_id)]   = {
            "filename":  filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks":    len(chunk_texts),
        }
        return _THREAD_METADATA[str(thread_id)].copy()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

# ─── Tools ──────────────────────────────────────────────────────────────────────

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def random_float_range(min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Generate a random float within a specified range."""
    return min_val + (max_val - min_val) * random.random()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    context = retriever.invoke(query)
    return {
        "query":       query,
        "context":     context,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools     = [search_tool, random_float_range, rag_tool]
_TOOL_MAP = {t.name: t for t in tools}

# ─── Tool schema → OpenAI format ────────────────────────────────────────────────

def _lc_tools_to_openai(lc_tools):
    return [
        {
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description,
                "parameters":  t.args_schema.schema() if t.args_schema
                               else {"type": "object", "properties": {}},
            },
        }
        for t in lc_tools
    ]

_OPENAI_TOOLS = _lc_tools_to_openai(tools)

# ─── Message conversion ──────────────────────────────────────────────────────────

def _to_openai_messages(messages: list) -> list:
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    result   = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
            continue
        role = role_map.get(getattr(m, "type", ""), "user")
        if isinstance(m, ToolMessage):
            result.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": str(m.content)})
        elif isinstance(m, AIMessage) and m.tool_calls:
            result.append({
                "role":       "assistant",
                "content":    m.content or "",
                "tool_calls": [
                    {
                        "id":   tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                    }
                    for tc in m.tool_calls
                ],
            })
        else:
            result.append({"role": role, "content": m.content or ""})
    return result

# ─── LLM call ───────────────────────────────────────────────────────────────────

def invoke_model(messages: list):
    if not messages:
        raise ValueError("messages cannot be empty")
    return client2.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=messages,
        tools=_OPENAI_TOOLS,
        tool_choice="auto",
    ).choices[0].message

# ─── Agentic tool loop ───────────────────────────────────────────────────────────

def run_with_tools(messages: list, thread_id: Optional[str] = None) -> AIMessage:
    plain = _to_openai_messages(messages)

    while True:
        response = invoke_model(plain)

        if not response.tool_calls:
            return AIMessage(content=response.content or "")

        plain.append({
            "role":       "assistant",
            "content":    response.content or "",
            "tool_calls": [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in response.tool_calls
            ],
        })

        for tc in response.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tc.function.name == "rag_tool" and "thread_id" not in args:
                args["thread_id"] = thread_id

            lc_tool = _TOOL_MAP.get(tc.function.name)
            result  = lc_tool.invoke(args) if lc_tool else f"Unknown tool: {tc.function.name}"

            plain.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      str(result),
            })

# ─── LangGraph ───────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState, config=None) -> dict:
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            f"the `rag_tool` and pass thread_id=`{thread_id}`. "
            "You can also use the web search tool and random_float_range when helpful. "
            "If no document is available and the user asks about one, ask them to upload a PDF."
        )
    )
    return {"messages": [run_with_tools([system, *state["messages"]], thread_id=thread_id)]}


conn         = sqlite3.connect(database="cb.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
cb = graph.compile(checkpointer=checkpointer)

# ─── Thread helpers ──────────────────────────────────────────────────────────────

def retrieve_all_threads() -> list:
    seen = set()
    for checkpoint in checkpointer.list(None):
        seen.add(checkpoint.config["configurable"]["thread_id"])
    return list(seen)


def get_first_human_message_10_words(thread_id: str) -> str:
    state = cb.get_state({"configurable": {"thread_id": thread_id}})
    if not state or "messages" not in state.values:
        return str(thread_id)
    for m in state.values["messages"]:
        if isinstance(m, HumanMessage):
            return " ".join(m.content.split()[:10])
        if isinstance(m, dict) and m.get("role") == "user":
            return " ".join(m["content"].split()[:10])
    return str(thread_id)