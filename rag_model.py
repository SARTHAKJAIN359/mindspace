import os
import json
import pickle
import re
import numpy as np
import faiss
from textblob import TextBlob

DATA_DIR = "data"
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")
RAG_FILE = os.path.join(DATA_DIR, "mental_health_knowledge_v2.json")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss_index.bin")
EMBEDDINGS_PATH = os.path.join(ARTIFACT_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.pkl")

EMBEDDER = None
FAISS_INDEX = None
RAG_DATA = None
DOCS = None


# ---------------------------
# LAZY-LOAD EMBEDDER
# ---------------------------
def get_embedder():
    """Lazy-load the SentenceTransformer model on first use.
    Avoids loading ~300MB of PyTorch into memory during gunicorn boot."""
    global EMBEDDER
    if EMBEDDER is None:
        print("[LAZY] Loading local embedding model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
        print("[LAZY] Embedding model loaded.")
    return EMBEDDER


# ---------------------------
# NORMALIZE
# ---------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# SPELL CORRECTION
# ---------------------------
def correct_spelling(text):
    try:
        if len(text.split()) <= 5:
            return str(TextBlob(text).correct())
        return text
    except:
        return text


# ---------------------------
# LOAD DATASET
# ---------------------------
def load_rag_corpus():
    with open(RAG_FILE, "r", encoding="utf-8") as f:
        rag = json.load(f)

    docs = []

    for entry in rag:
        q = entry.get("question", "")
        alts = " ".join(entry.get("alternate_questions", []))
        keywords = " ".join(entry.get("keywords", []))
        a = entry.get("answer", "")

        text = normalize(f"{q} {alts} {keywords} {entry.get('category', '')} {entry.get('intent_type', '')} {a}")
        docs.append(text)

    return rag, docs


# ---------------------------
# INIT RAG (FAISS + LOCAL EMBEDDINGS)
# ---------------------------
def init_rag():
    """Initialise FAISS index and (optionally) local sentence-transformer embeddings.
    If pre-built artifacts exist, the embedder is NOT loaded at startup — it will
    be lazy-loaded on the first query to keep peak boot memory under 512 MB."""
    global FAISS_INDEX, RAG_DATA, DOCS
    RAG_DATA, DOCS = load_rag_corpus()

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
            if FAISS_INDEX.d != 384:
                raise ValueError("Dimension mismatch")
            print("FAISS index loaded from pre-built artifacts.")
            return  # Embedder will be lazy-loaded on first query
        except:
            print("Warning: Rebuilding FAISS index due to dimension mismatch or error...")
            FAISS_INDEX = None

    if FAISS_INDEX is None:
        print("Generating local embeddings for RAG... This will take a few seconds.")
        embedder = get_embedder()
        
        emb_matrix = embedder.encode(DOCS, convert_to_numpy=True)
        faiss.normalize_L2(emb_matrix)
        
        dim = emb_matrix.shape[1]
        FAISS_INDEX = faiss.IndexFlatIP(dim)
        FAISS_INDEX.add(emb_matrix)
        
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
        
        # Save raw embeddings and metadata (the 2 extra files)
        np.save(EMBEDDINGS_PATH, emb_matrix)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(RAG_DATA, f)
            
        print("FAISS index, embeddings, and metadata built and saved locally.")


# ---------------------------
# QUERY EXPANSION
# ---------------------------
def expand_query(query):
    synonyms = {
        "sad": "depressed hopeless",
        "anxious": "anxiety panic worry",
        "stress": "pressure overwhelmed burnout",
        "tired": "fatigue exhausted",
        "overthinking": "rumination intrusive thoughts",
        "lonely": "alone isolated",
        "panic": "panic attack anxiety",
        "empty": "depression numb"
    }

    expanded = query.lower()
    for k, v in synonyms.items():
        if k in expanded:
            expanded += " " + v

    return expanded


# ---------------------------
# CATEGORY INTENT
# ---------------------------
def detect_intent(query):
    q = query.lower()

    intent_map = {
        "stress": ["stress", "overwhelmed", "pressure"],
        "anxiety": ["anxiety", "panic", "fear"],
        "depression": ["depressed", "sad", "hopeless"],
        "relationships": ["breakup", "partner"],
        "self-esteem": ["confidence", "insecure"]
    }

    for intent, words in intent_map.items():
        if any(w in q for w in words):
            return intent

    return None


# ---------------------------
# INTENT TYPE (NEW)
# ---------------------------
def detect_intent_type(query):
    q = query.lower()

    if any(x in q for x in ["how to", "cope", "deal", "manage", "reduce"]):
        return "coping"
    if any(x in q for x in ["what is", "meaning"]):
        return "definition"
    if any(x in q for x in ["why", "cause"]):
        return "cause"
    if any(x in q for x in ["symptoms", "signs"]):
        return "symptoms"

    return "emotional"


# ---------------------------
# MAIN QUERY
# ---------------------------
def answer_query(query, top_k=3):
    corrected = correct_spelling(query)

    intent = detect_intent(corrected)
    intent_type = detect_intent_type(corrected)

    expanded = expand_query(corrected)
    normalized_q = normalize(expanded)
    
    # Generate query embedding locally
    query_emb = get_embedder().encode([normalized_q], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    # Search FAISS (fetch more to allow filtering)
    fetch_k = max(top_k * 3, 10)
    faiss_scores, faiss_indices = FAISS_INDEX.search(query_emb, fetch_k)

    results = []
    q_lower = corrected.lower()

    for i, idx in enumerate(faiss_indices[0]):
        if idx == -1: continue
        entry = RAG_DATA[int(idx)]

        raw_score = float(faiss_scores[0][i])
        score = raw_score * 1.5

        category = entry.get("category", "")
        keywords = entry.get("keywords", [])
        alts = entry.get("alternate_questions", [])
        entry_intent_type = entry.get("intent_type")

        # ---------------------------
        # FILTERS
        # ---------------------------
        if intent and category != intent:
            continue

        if intent_type and entry_intent_type != intent_type:
            continue

        # ---------------------------
        # BOOSTS
        # ---------------------------

        # keyword match
        if any(k in q_lower for k in keywords):
            score += 0.05

        # category boost
        if intent and category == intent:
            score += 0.15

        # intent_type boost (VERY IMPORTANT)
        if intent_type == entry_intent_type:
            score += 0.2

        # alternate question boost
        if any(q_lower in alt.lower() or alt.lower() in q_lower for alt in alts):
            score += 0.15

        score = min(score, 1.0)

        results.append({
            "id": entry.get("id"),
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "category": category,
            "score": round(score, 3)
        })

    # ---------------------------
    # FALLBACK
    # ---------------------------
    if not results:
        for i, idx in enumerate(faiss_indices[0][:top_k]):
            if idx == -1: continue
            entry = RAG_DATA[int(idx)]
            results.append({
                "id": entry.get("id"),
                "question": entry.get("question"),
                "answer": entry.get("answer"),
                "category": entry.get("category"),
                "score": round(float(faiss_scores[0][i]), 3)
            })

    # Limit to top_k after filters
    results = results[:top_k]

    # ---------------------------
    # SMART FUSION
    # ---------------------------
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    if len(results) > 1 and abs(results[0]["score"] - results[1]["score"]) < 0.04:
        results[0]["answer"] += " " + results[1]["answer"]

    return results


# ---------------------------
# FILTER
# ---------------------------
def filter_results(results, threshold=0.18):
    return [r for r in results if r["score"] >= threshold]


def save_embeddings():
    return