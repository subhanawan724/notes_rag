


from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from tavily import TavilyClient
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
import os

class HFInferenceEmbeddings:
    def __init__(self, api_token, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(token=api_token, model=model_name)

    def embed_query(self, text):
        return self.client.feature_extraction(text).tolist()

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


 

embeddings = HFInferenceEmbeddings(api_token=os.environ["HF_API_TOKEN"])

db = Chroma(
    persist_directory="data/notes_vector_db",
    embedding_function=embeddings,
)



client = InferenceClient(token=os.environ["HF_API_TOKEN"], provider="featherless-ai")



tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
#tavily = TavilyClient(api_key=TAVILY_API_KEY)

def ask_my_ai_bot(user_query, db):
    retrieved = db.similarity_search_with_score(user_query, k=1)
    doc, score = retrieved[0]
   # print(f"QUERY: {user_query} | SCORE: {score}")
    raw_csv_text = doc.page_content

    SCORE_THRESHOLD = 1.2 

    if score > SCORE_THRESHOLD:
       
        search_results = tavily.search(query=user_query)
        web_snippets = "\n".join([r["content"] for r in search_results["results"]])
        source_note = "This question wasn't found in the local notes — using web search results instead."
        reference_text = web_snippets
    else:
       
        source_note = "Answered using local course notes."
        reference_text = raw_csv_text

    messages = [
        {"role": "system", "content": "You are an expert AI Programming Assistant. Explain the user's question using ONLY the provided reference documentation."},
        {"role": "user", "content": f"[{source_note}]\n[REFERENCE]:\n{reference_text}\n\n[QUESTION]:\n{user_query}"}
    ]

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return completion.choices[0].message.content



app = FastAPI()



class QueryModel(BaseModel):
    user_query: str

@app.post("/question")
def question(data: QueryModel):
    answer = ask_my_ai_bot(data.user_query, db)
    return {"answer": answer}


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")
