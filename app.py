from flask import Flask, request, jsonify, render_template
import os
from groq import Groq
from rag_model import answer_query, filter_results, init_rag
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK", "key")


GROQ_KEYS = [
    os.environ.get("GROQ_API_KEY_1"),
    os.environ.get("GROQ_API_KEY_2"),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]  # drop missing / None

LLM_MODEL = "openai/gpt-oss-20b"

# Track which key is currently active (index into GROQ_KEYS)
_key_index = 0


def get_llm_client():
    """Return a Groq client for the current active key."""
    if not GROQ_KEYS:
        return None
    return Groq(api_key=GROQ_KEYS[_key_index])


def rotate_key():
    """Switch to the next available Groq key (called on rate-limit errors)."""
    global _key_index
    if len(GROQ_KEYS) > 1:
        _key_index = (_key_index + 1) % len(GROQ_KEYS)
        print(f"[ROTATE] Rotated to Groq key index {_key_index}")


def llm_chat(prompt: str, max_tokens: int = 300) -> str:
    """
    Call the Groq LLM with automatic key rotation on rate-limit errors.
    Keeps max_tokens conservative to protect the TPM/TPD budget.
    """
    last_error = None
    for attempt in range(len(GROQ_KEYS) * 2):          
        client = get_llm_client()
        if client is None:
            break
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # Rate-limit / quota exhausted → try the other key
            if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                print(f"[WARN] Rate limit on key {_key_index}: {e}")
                rotate_key()
            else:
                print(f"[ERR] LLM error: {e}")
                raise  # non-rate-limit errors bubble up immediately

    raise RuntimeError(f"All Groq keys exhausted. Last error: {last_error}")


if GROQ_KEYS:
    print(f"[OK] Groq LLM enabled ({LLM_MODEL}) with {len(GROQ_KEYS)} key(s)")
else:
    print("[WARN] Groq LLM not enabled -- set GROQ_API_KEY_1 / GROQ_API_KEY_2 in .env")


# ---------------------------
# PREPARE RAG (FAISS + Local Embeddings)
# ---------------------------
try:
    init_rag()
except Exception as e:
    print("RAG init error:", e)


# ---------------------------
# CRISIS KEYWORDS
# ---------------------------
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "i want to die", "i wanna die", "i don't want to live",
    "better off dead", "no reason to live", "can't go on",
    "give up", "overdose", "cut myself", "hurt myself",
    "harm myself", "i feel like ending it", "life is pointless",
    "i am done with life", "i can't do this anymore",
    "i feel like disappearing", "i wish i was dead"
]


# ---------------------------
# HELPLINES
# ---------------------------
HELPLINES = {
    "india": "Tele-MANAS: 14416",
    "usa": "988 Suicide & Crisis Lifeline",
    "uk": "Samaritans: 116 123",
    "canada": "Talk Suicide Canada: 1-833-456-4566",
    "australia": "Lifeline: 13 11 14",
    "germany": "TelefonSeelsorge: 0800 1110 111",
    "france": "Suicide Écoute: 01 45 39 40 00",
    "brazil": "Centro de Valorização da Vida: 188",
    "japan": "Yorisoi Hotline: 0120-279-338",
    "south africa": "SADAG: 0800 567 567",
    "pakistan": "Umang Pakistan: 0311-7786264",
    "bangladesh": "Kaan Pete Roi: 09639 678 999",
    "nepal": "TPO Nepal: 1660-01-22211",
    "sri lanka": "Sumithrayo: 011 269 6666"
}


def get_helpline(country):
    if not country:
        return HELPLINES["india"]

    country = country.lower().strip()

    mapping = {
        "us": "usa", "united states": "usa",
        "gb": "uk", "uk": "uk",
        "in": "india", "ca": "canada",
        "au": "australia", "de": "germany",
        "fr": "france", "br": "brazil",
        "jp": "japan", "za": "south africa",
        "pk": "pakistan", "bd": "bangladesh",
        "np": "nepal", "lk": "sri lanka"
    }

    if country == "other":
        return "Please contact local emergency services."

    country = mapping.get(country, country)
    return HELPLINES.get(country, HELPLINES["india"])


# ---------------------------
# QUERY REWRITING
# ---------------------------
def rewrite_query(user_input, history_text):
    """Rewrite conversational query into a standalone search query via LLM."""
    if not history_text or not GROQ_KEYS:
        return user_input

    prompt = (
        "Given the following chat history and the user's latest message, "
        "rewrite the user's message into a standalone search query that can be "
        "understood without context.\n"
        "Do not answer the question. Only return the rewritten query. "
        "If it doesn't need rewriting, return the original message.\n\n"
        f"{history_text}"
        f"User's latest message: {user_input}\n"
        "Rewritten search query:"
    )

    try:
        rewritten = llm_chat(prompt, max_tokens=80)
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        print(f"[REWRITE] Query Rewritten: '{user_input}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        print("Query rewrite error:", e)
        return user_input


# ---------------------------
# ANSWER ENRICHMENT
# ---------------------------
def enrich_answer(raw_answer, user_input):
    """Add empathy/warmth to a raw KB answer using the LLM."""
    if not GROQ_KEYS:
        return raw_answer

    prompt = (
        "Expand this mental health support answer to be more empathetic and "
        "conversational in 2-3 sentences. Do not add new facts or diagnose anything. "
        "Keep it warm and supportive.\n"
        f"Original Answer: {raw_answer}\n"
        f"User asked: {user_input}\n"
        "Expanded empathetic response:"
    )

    try:
        return llm_chat(prompt, max_tokens=200)
    except Exception as e:
        print("Enrich answer error:", e)
        return raw_answer


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("input") or "").strip()
    selected_mode = data.get("mode", "hybrid")
    country = (data.get("country") or "India").strip()
    history = data.get("history", [])

    history_text = ""
    if history:
        history_text = "Chat History:\n" + "\n".join(
            [f"{msg['role'].capitalize()}: {msg['text']}" for msg in history[-5:]]
        ) + "\n\n"

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    lower = user_input.lower()

    # ---------------------------
    # CRISIS DETECTION
    # ---------------------------
    if any(k in lower for k in CRISIS_KEYWORDS):
        helpline = get_helpline(country)
        return jsonify({
            "response": (
                f"I'm really sorry you're feeling this way. You're not alone.\n\n"
                f"Please consider reaching out immediately to {helpline} or someone you trust.\n\n"
                "If you're in immediate danger, please contact local emergency services."
            ),
            "is_crisis": True
        })

    # ---------------------------
    # CONTEXTUAL QUERY REWRITING
    # ---------------------------
    search_query = rewrite_query(user_input, history_text)

    # ---------------------------
    # IR MODEL MODE (pure Knowledge Base retrieval — no LLM)
    # ---------------------------
    if selected_mode == "ir":
        results = answer_query(search_query, top_k=3)
        filtered = filter_results(results, threshold=0.16)

        if not filtered:
            return jsonify({
                "response": "I'm not fully sure, but it sounds important. Consider talking to someone you trust.",
                "mode": "ir"
            })

        best = filtered[0]

        # Enrich the raw KB answer with empathy via LLM
        final_answer = enrich_answer(best["answer"], user_input)

        return jsonify({
            "response": final_answer,
            "confidence": best["score"],
            "category": best["category"],
            "mode": "ir",
            "is_crisis": (best["category"] == "crisis")
        })

    # ---------------------------
    # LLM MODE (Llama via Groq — pure conversational)
    # ---------------------------
    elif selected_mode == "llm":
        if not GROQ_KEYS:
            return jsonify({"response": "LLM not available — API keys not configured.", "mode": "error"})

        helpline = get_helpline(country)

        prompt = (
            "You are MindSpace, a compassionate mental health assistant.\n\n"
            "Be empathetic and supportive. Do not diagnose.\n\n"
            f"If the user shows distress, gently suggest help: {helpline}\n\n"
            f"{history_text}User: {user_input}\n"
            "Response:"
        )

        try:
            reply = llm_chat(prompt, max_tokens=350)
            return jsonify({"response": reply, "mode": "llm"})
        except Exception as e:
            return jsonify({"response": str(e), "mode": "error"})

    # ---------------------------
    # HYBRID MODE — IRIS (RAG + LLM combined)
    # ---------------------------
    else:
        results = answer_query(search_query, top_k=3)
        filtered = filter_results(results, threshold=0.16)

        best = filtered[0] if filtered else None

        context = "\n\n".join([
            f"Q: {r['question']}\nA: {r['answer']}\nCategory: {r['category']}"
            for r in filtered[:2]
        ]) if filtered else ""

        helpline = get_helpline(country)

        # ✅ CASE 1: Strong KB match → Hybrid (LLM enriches KB answer)
        if GROQ_KEYS and context:
            try:
                prompt = (
                    "You are MindSpace, a warm and compassionate mental health support assistant.\n\n"
                    "INSTRUCTIONS:\n"
                    "- Use the Knowledge Base below as your primary factual source. Do not contradict it.\n"
                    "- Expand it naturally with empathy, warmth, and a conversational tone.\n"
                    "- Do NOT diagnose any medical condition.\n"
                    "- Acknowledge the user's feelings before providing information.\n"
                    "- Keep the response concise but warm (2-4 sentences).\n\n"
                    f"Knowledge Base:\n{context}\n\n"
                    f"{history_text}User: {user_input}\n\n"
                    "Response:"
                )
                reply = llm_chat(prompt, max_tokens=350)
                return jsonify({
                    "response": reply,
                    "confidence": best["score"] if best else None,
                    "category": best["category"] if best else None,
                    "mode": "hybrid",
                    "rag_used": True,
                    "is_crisis": (best["category"] == "crisis") if best else False
                })
            except Exception as e:
                print("LLM hybrid error:", e)

        # ✅ CASE 2: Weak/No KB match → LLM fallback
        if GROQ_KEYS:
            try:
                prompt = (
                    "You are MindSpace, a supportive mental health assistant.\n\n"
                    "Be helpful, empathetic, and conversational.\n"
                    "Do not diagnose medical conditions.\n\n"
                    f"If user shows distress, suggest help: {helpline}\n\n"
                    f"{history_text}User: {user_input}\n\n"
                    "Response:"
                )
                reply = llm_chat(prompt, max_tokens=350)
                return jsonify({"response": reply, "mode": "llm_fallback"})
            except Exception as e:
                print("LLM fallback error:", e)

        # FINAL fallback → raw KB answer
        if best:
            return jsonify({
                "response": best["answer"],
                "confidence": best["score"],
                "category": best["category"],
                "mode": "fallback",
                "is_crisis": (best["category"] == "crisis")
            })

        return jsonify({
            "response": "I'm not sure, but you can try rephrasing.",
            "mode": "fallback"
        })


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
    