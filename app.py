import os
import threading
import faiss
import numpy as np
import pandas as pd
import json
import re
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

import time

# ---------------------------
# Config & Silence Warnings
# ---------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
CSV_PATH = os.environ.get("RAG_CSV_PATH", os.path.join("data", "Pokemon.csv"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ---------------------------
# Prompts
# ---------------------------
EXPERT_PROMPT_TEMPLATE = """
You are a Pokemon Data Expert. 

GUIDELINES:
1. Primary Source: Always prioritize answering the user's question based ONLY on the provided Context. Use the Context for all numerical stats (HP, Attack, etc.).
2. Internal Knowledge Exception: If the user asks for background, lore, or descriptive information about a SPECIFIC Pokemon (or specific Pokemons) AND the Context does not contain this information, you are allowed to use your internal knowledge to provide the background strictly about that requested Pokemon.
3. Strict Limitation: Do NOT use internal knowledge for mathematical aggregations (e.g., "How many Gen 2 Pokemon exist?") or for questions unrelated to Pokemon.
4. Unknowns: If the question is not about a specific Pokemon's background, and the answer cannot be found in the context, state clearly: "I do not have enough information in the retrieved context to answer this."

Context:
{context}

Question: {question}
Answer:"""

# ---------------------------
# Flask + globals
# ---------------------------
app = Flask(__name__)
os.makedirs(DATA_DIR, exist_ok=True)

_lock = threading.Lock()
_docs = []
_df = None  # We will keep the original DataFrame in memory for Pandas
_index = None
_embed_model = None
_gemini_client = None


# ---------------------------
# Helper Functions
# ---------------------------
def _safe_extract_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M)
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# ---------------------------
# Data Loading & Indexing
# ---------------------------
def load_pokemon_data():
    global _df
    if not os.path.exists(CSV_PATH):
        return ["Error: data/Pokemon.csv not found."]

    _df = pd.read_csv(CSV_PATH)
    all_texts = []
    for _, row in _df.iterrows():
        t2 = f", Type 2: {row['Type 2']}" if pd.notna(row['Type 2']) else ""
        desc = (f"Name: {row['Name']} | Type 1: {row['Type 1']}{t2} | "
                f"Total: {row['Total']}, HP: {row['HP']}, Attack: {row['Attack']}, "
                f"Defense: {row['Defense']}, Sp. Atk: {row['Sp. Atk']}, Sp. Def: {row['Sp. Def']}, Speed: {row['Speed']} | "
                f"Gen: {row['Generation']}, Legendary: {row['Legendary']}")
        all_texts.append(desc)
    return all_texts


def rebuild_index():
    global _docs, _index, _embed_model
    with _lock:
        if _embed_model is None:
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _docs = load_pokemon_data()
        if _docs and not _docs[0].startswith("Error"):
            embeddings = _embed_model.encode(_docs, normalize_embeddings=True)
            _index = faiss.IndexFlatIP(embeddings.shape[1])
            _index.add(np.array(embeddings).astype('float32'))


# ---------------------------
# Agentic RAG: Router & Pandas
# ---------------------------
def route_query(query):
    # A router that decides whether the query is analytical (Pandas) or semantic (FAISS)
    client = get_gemini_client()
    prompt = f"""You are a routing agent for a Pokemon database.
Classify the user's question into one of two routes:
- "pandas": if the question requires math, counting, finding maximums/minimums, or strict statistics (e.g., "highest HP", "how many fire types").
- "faiss": if the question asks for descriptions, general info, or specific traits of a named Pokemon (e.g., "what are Charizard's types?", "who is Pikachu").

Question: "{query}"
Return ONLY JSON: {{"route": "pandas"}} or {{"route": "faiss"}}"""

    try:
        res = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = _safe_extract_json(res.text)
        return data.get("route", "faiss").lower()
    except:
        return "faiss"  # return "faiss" default in case of error


def execute_pandas_route(query):
    #Generates and runs Pandas code instead of using vector search
    client = get_gemini_client()
    cols = list(_df.columns)

    prompt = f"""You are a Python Pandas expert.
I have a DataFrame 'df' with columns: {cols}.
Write ONLY a single Python expression using 'df' that evaluates to the answer for the question.
Do not use print(). Do not use markdown. Just the code.

Examples:
Q: How many Gen 2 pokemons?
Code: df[df['Generation'] == 2].shape[0]

Q: Which pokemon has the highest Attack?
Code: df.loc[df['Attack'].idxmax()]['Name']

Question: {query}
Code:"""

    res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    code_expr = res.text.strip().replace("```python", "").replace("```", "").strip()

    try:
        # הרצת הקוד על ה-DataFrame שלנו
        result = eval(code_expr, {"df": _df, "pd": pd, "np": np})
        return f"Pandas Code Executed: {code_expr}\nRaw Result: {result}", [
            f"Extracted directly from DataFrame using: {code_expr}"]
    except Exception as e:
        print(f"Pandas Eval Error: {e}")
        return None, []


# ---------------------------
# Retrieval (FAISS)
# ---------------------------
def retrieve_faiss(query, k=5):
    # Simple and easy vector retrieval (we returned k to be small and fast)
    q_emb = _embed_model.encode([query], normalize_embeddings=True).astype('float32')
    _, I = _index.search(q_emb, k)
    return [_docs[i] for i in I[0]]


# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.post("/ask")
def ask():
    start_time = time.time()  # Start timer
    api_requests_count = 0  # Counter for API calls

    data = request.get_json() or {}
    q = data.get("question", "").strip()
    if not q: return jsonify({"error": "No question provided"}), 400

    with _lock:
        if not _docs or _index is None:
            return jsonify({"error": "No documents indexed."}), 500

        # 1. Query Routing
        route = route_query(q)
        api_requests_count += 1  # 1 request for routing
        print(f"🧠 Routing '{q}' -> {route.upper()} Route")

        context_str = ""
        context_sources = []

        # 2. Path Selection
        if route == "pandas":
            context_str, context_sources = execute_pandas_route(q)
            api_requests_count += 1  # 1 request for generating pandas code

        if not context_str or route == "faiss":
            context_sources = retrieve_faiss(q, k=5)
            context_str = "\n".join(context_sources)

    # 3. Final Answer Generation
    client = get_gemini_client()
    final_prompt = EXPERT_PROMPT_TEMPLATE.format(context=context_str, question=q)

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=final_prompt)
        api_requests_count += 1  # 1 request for the final answer
        answer = response.text

        # Extract token usage from Gemini response
        tokens_used = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_used = response.usage_metadata.total_token_count
    except Exception as e:
        answer = f"Error generating answer: {e}"
        tokens_used = "Error"

    exec_time = round(time.time() - start_time, 2)  # Calculate total seconds

    return jsonify({
        "answer": answer,
        "context": context_sources,
        "question": q,
        "route_used": route.upper(),
        "tokens": tokens_used,
        "api_requests": api_requests_count,
        "exec_time": exec_time
    })

if __name__ == "__main__":
    rebuild_index()
    app.run(host="0.0.0.0", port=5001, debug=True)