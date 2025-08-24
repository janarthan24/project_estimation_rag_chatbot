import sqlite3
import os
import pandas as pd
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import login


# ----------------------------
# Config
# ----------------------------
INDEX_PATH = "estimation_faiss.index"
TEXTS_PATH = "faiss_texts.pkl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLAMA_ENDPOINT = "http://localhost:11434/v1/chat/completions"  # local LLaMA API (Ollama/vLLM/llama-cpp)
# Local LLM (Ollama)
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama2:7b"  # <- per your request

# ----------------------------
# Utility: Build FAISS Index
# ----------------------------
HF_TOKEN = "dummy"  # replace with your actual token or set as env variable
login(token=HF_TOKEN)
embedder = SentenceTransformer(EMBED_MODEL,token=HF_TOKEN,trust_remote_code=True,device="cpu")
reranker = CrossEncoder(RERANK_MODEL,token=HF_TOKEN,trust_remote_code=True,device="cpu")
    
def build_faiss_index(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.prj_name,s.prj_type, s.domain, s.type_of_estimation, s.month, d.feature_or_scenarios,
               d.technology_or_type, d.module_or_method, d.complexity, d.total_SP, d.type
        FROM estimation_summary s
        JOIN estimation_details d ON s.id = d.parent_id
    """)
    rows = cur.fetchall()
    conn.close()

    docs = []
    for r in rows:
        pid, pname,ptype, domain, etype, month, feat, tech, module, comp, sp, t = r
        content = f"""
        Project: {pname}
        Project Type: {ptype}
        Domain: {domain}
        Estimation Type: {etype}
        Month: {month}
        Feature/Scenario: {feat}
        Technology: {tech}
        Module: {module}
        Complexity: {comp}
        Story Points: {sp}
        Type: {t}
        """
        docs.append({
            "text": content.strip(),
            "meta": {"parent_id": pid}
        })

    embeddings = embedder.encode([d["text"] for d in docs], show_progress_bar=True, convert_to_numpy=True)


    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(docs, f)

    return len(docs)


# ----------------------------
# Utility: Retrieve with Reranker
# ----------------------------
def retrieve_with_rerank(query, top_k=5):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(TEXTS_PATH):
        build_faiss_index()

    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        docs = pickle.load(f)

    print("Loaded FAISS index with", len(docs), "documents.")

    
    query_emb = embedder.encode([query], convert_to_numpy=True)

    D, I = index.search(query_emb, top_k * 3)  # over-retrieve for reranking
    candidates = [docs[i] for i in I[0] if i < len(docs)]


    scores = reranker.predict([(query, c["text"]) for c in candidates])


    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in reranked[:top_k]]

def build_prompt(query: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        [f"[Doc {i+1}] (parent_id={c['meta'].get('parent_id')})\n{c['text']}"
         for i, c in enumerate(contexts)]
    )
    prompt = f"""You are an enterprise estimation assistant.
Use ONLY the context below to answer. If the answer cannot be found, say you don't have enough
information and suggest what additional data is needed.

# Context
{context_block}

# User Question
{query}

# Output Guidelines
- Keep the answer **short (3–4 sentences maximum)**.
- If you cite data, reference the [Doc #].
- If numbers are summed, show a quick calculation line.
- If you infer, clearly label it as an inference.

Answer:
"""
    return prompt

def call_ollama(prompt: str, temperature: float = 0.2, max_tokens: int = 120) -> str:
 url = f"{OLLAMA_HOST}/api/chat"
 payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are an enterprise estimation assistant."},
            {"role": "user", "content": prompt}
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }
 try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
 except Exception as e:
        return f"[LLM error: {e}]"

# ----------------------------
# Utility: Call Local LLaMA
# ----------------------------
def call_llama(query, context):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama2",  # adjust if your Ollama/vLLM model name differs
        "messages": [
            {"role": "system", "content": "You are an enterprise assistant helping with estimation dashboard data."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        "temperature": 0.3,
        "max_tokens": 120
    }

    try:
        resp = requests.post(LLAMA_ENDPOINT, headers=headers, json=payload)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error contacting LLaMA: {e}"

# ----------------------------