from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from tavily import TavilyClient
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
import os

# --- NAYA ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
# --- NAYA khatam ---

class HFInferenceEmbeddings:
    def __init__(self, api_token, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(token=api_token, model=model_name)

    def embed_query(self, text):
        return self.client.feature_extraction(text).tolist()

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

embeddings = HFInferenceEmbeddings(api_token=os.environ["HF_API_TOKEN"])

db = Chroma(
    persist_directory=r"D:\Courses_lectures\data\notes_vector_db",
    embedding_function=embeddings,
)

client = InferenceClient(token=os.environ["HF_API_TOKEN"], provider="featherless-ai")

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

SCORE_THRESHOLD = 1.2

# ============================================================
# NAYA — LangGraph state, nodes, aur wiring
# ============================================================

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    critique: str
    needs_more_research: bool
    revision_number: int
    max_revisions: int
    history: Annotated[list, operator.add]


def retrieve_node(state: AgentState):
    retrieved = db.similarity_search_with_score(state["question"], k=1)
    doc, score = retrieved[0]
    context = doc.page_content if score < SCORE_THRESHOLD else ""
    return {"context": context, "revision_number": 1, "max_revisions": 2}


def generate_node(state: AgentState):
    history_text = ""
    for turn in state.get("history", [])[-3:]:
        history_text += f"Previous Q: {turn['question']}\nPrevious A: {turn['answer']}\n\n"

    messages = [
        {"role": "system", "content": "You are an expert AI Programming Assistant . Explain thoroughly using ONLY the provided reference/context. Whenever you explain a concept or syntax, include a short code example, even if not explicitly asked."},
        {"role": "user", "content": f"{history_text}[REFERENCE]:\n{state['context']}\n\n[QUESTION]:\n{state['question']}"}
    ]
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return {"answer": completion.choices[0].message.content}


CRITIQUE_PROMPT = """You are a strict teacher reviewing an AI-generated answer .
Check: (1) Is it too short or vague? (2) Does it need a code example but is missing one? (3) Does it fully use the available context?
If everything is good, respond with exactly: GOOD
If not, respond with a short list of what's missing, phrased as a search query to find more information."""

def critique_node(state: AgentState):
    messages = [
        {"role": "system", "content": CRITIQUE_PROMPT},
        {"role": "user", "content": f"Question: {state['question']}\n\nAnswer: {state['answer']}\n\nContext used: {state['context']}"}
    ]
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        messages=messages,
        max_tokens=150,
        temperature=0.2,
    )
    critique = completion.choices[0].message.content
    needs_more = critique.strip() != "GOOD"
    return {"critique": critique, "needs_more_research": needs_more}


def research_more_node(state: AgentState):
    search_results = tavily.search(query=f"{state['question']} {state['critique']}")
    web_snippets = "\n".join([r["content"] for r in search_results["results"]])
    new_context = state["context"] + "\n\n" + web_snippets
    return {"context": new_context, "revision_number": state["revision_number"] + 1}


def finalize_node(state: AgentState):
    return {"history": [{"question": state["question"], "answer": state["answer"]}]}


def should_continue(state: AgentState):
    if not state["needs_more_research"]:
        return "finalize"
    if state["revision_number"] >= state["max_revisions"]:
        return "finalize"
    return "research_more"


builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("critique", critique_node)
builder.add_node("research_more", research_more_node)
builder.add_node("finalize", finalize_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges("critique", should_continue, {"finalize": "finalize", "research_more": "research_more"})
builder.add_edge("research_more", "generate")
builder.add_edge("finalize", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Filhal single, shared conversation — sabhi users isi thread mein
THREAD = {"configurable": {"thread_id": "default"}}

# ============================================================
# App / endpoint — updated
# ============================================================

app = FastAPI()


class QueryModel(BaseModel):
    user_query: str


@app.post("/question")
def question(data: QueryModel):
    result = graph.invoke({"question": data.user_query}, THREAD)
    return {"answer": result["answer"]}


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")