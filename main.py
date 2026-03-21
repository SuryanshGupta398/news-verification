from fastapi import FastAPI, APIRouter, Form
import joblib
from pymongo import MongoClient
import os

app = FastAPI()
news_router = APIRouter(prefix="/news", tags=["News"])
@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.head("/health")
def health_check_head():
    return {"status": "ok"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

from fastapi import HTTPException

# import numpy as np

# from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from datetime import datetime

# def utc_now_iso():
#     return datetime.utcnow().isoformat()

# @app.get("/health")
# def health_check():
#     return {"status": "ok", "time": utc_now_iso()}

import threading

model = None
vectorizer = None
news_docs = []
tfidf_db = None

def load_resources_background():
    global model, vectorizer, news_docs, tfidf_db

    try:
        print("🔹 Loading model...")
        model = joblib.load("fake_news_model.pkl")

        print("🔹 Loading vectorizer...")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")

        print("🔹 Connecting MongoDB...")
        mongo_url = os.getenv("MONGO_URI")
        client = MongoClient(mongo_url)

        db = client["fake_new_app"]
        collection = db["news"]

        news_docs = list(collection.find({}, {"title":1,"description":1,"source":1}))

        print(f"✅ Loaded {len(news_docs)} news")

        if news_docs:
            texts = [
                n.get("title","") + " " + n.get("description","")
                for n in news_docs
            ]
            tfidf_db = vectorizer.transform(texts)

        print("✅ Background loading done")

    except Exception as e:
        print("❌ ERROR:", str(e))


@app.on_event("startup")
def startup_event():
    print("🚀 Server starting...")

    # Run heavy loading in background thread
    thread = threading.Thread(target=load_resources_background)
    thread.start()
    
class NewsRequest(BaseModel):
    text: str

# -----------------------------
# Find Similar News
# -----------------------------
def find_similar_news(user_text):
    if len(news_docs) == 0:
        return None, 0

    tfidf_user = vectorizer.transform([user_text])

    similarity = cosine_similarity(tfidf_user, tfidf_db)

    index = similarity.argmax()
    score = similarity[0][index]

    return news_docs[index], score
    
import re
from sklearn.metrics.pairwise import cosine_similarity

# ── Extract named entities (people, orgs) from text ───────────────
def extract_entities(text: str) -> set:
    """
    Simple regex-based NER — finds capitalized word sequences.
    Lightweight alternative to spaCy/NLTK.
    """
    text_clean = re.sub(r'[^\w\s]', ' ', text)
    # Match sequences of capitalized words (names, orgs)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text_clean)
    # Also grab common Hindi name patterns via known list
    return {e.lower() for e in entities}


# ── Sentiment polarity — no library needed ────────────────────────
# ── Polarity word lists ────────────────────────────────────────────
NEGATIVE_WORDS = {
    "died", "dead", "death", "killed", "murdered", "lost", "defeated",
    "arrested", "jailed", "fired", "resigned", "crashed", "failed",
    "attacked", "bombed", "expelled", "suspended", "banned", "collapsed",
    "injured", "hospitalized", "missing", "destroyed", "convicted",
    "mart", "mara", "maut", "hatya", "nahi", "gir", "toot",
}

POSITIVE_WORDS = {
    "won", "win", "victory", "launched", "inaugurated", "appointed",
    "elected", "promoted", "visited", "announced", "achieved", "grew",
    "increased", "recovered", "released", "celebrated", "awarded",
    "saved", "beat", "champion", "topped", "passed", "selected",
    "jeeta", "jeet", "vijay", "safal", "aaya", "khola", "mila",
}

STOP_WORDS = {
    "the","a","an","is","are","was","were","in","on","at","to","for",
    "of","and","or","with","this","that","it","by","from","has","have",
    "had","be","been","being","will","would","could","should","may",
    "might","do","does","did","not","no","its","our","their","we","he",
    "she","they","i","you","me","him","her","us","them","as","but","if",
    "than","so","yet","both","either","each","than","too","very","just",
}

def get_polarity(text: str) -> str:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    neg = len(words & NEGATIVE_WORDS)
    pos = len(words & POSITIVE_WORDS)
    if neg > pos: return "negative"
    if pos > neg: return "positive"
    return "neutral"

def word_overlap_score(a: str, b: str) -> float:
    """Overlap score with stop word removal."""
    a_words = set(re.findall(r'\b\w+\b', a.lower())) - STOP_WORDS
    b_words = set(re.findall(r'\b\w+\b', b.lower())) - STOP_WORDS
    if not a_words or not b_words:
        return 0.0
    overlap = a_words & b_words
    return len(overlap) / max(len(a_words), len(b_words))

def check_google_factcheck(claim: str) -> dict:
    """
    Searches Google Fact Check Tools API.
    No date restriction — checks all time.
    """
    GOOGLE_KEY = os.getenv("GOOGLE_FACT_CHECK_KEY")
    if not GOOGLE_KEY:
        return {"found": False, "reason": "GOOGLE_FACT_CHECK_KEY not set"}
    try:
        resp = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={
                "query":        claim,
                "key":          GOOGLE_KEY,
                "languageCode": "en",
                "pageSize":     5,        # get top 5 fact checks
            },
            timeout=6,
        )
        data = resp.json()

        if "claims" not in data or not data["claims"]:
            return {"found": False, "reason": "No fact-check records found"}

        results = []
        for item in data["claims"][:5]:
            for review in item.get("claimReview", []):
                rating = review.get("textualRating", "").lower()

                if any(w in rating for w in ["false","fake","misleading","incorrect","wrong","pants on fire","inaccurate"]):
                    verdict = "FAKE"
                elif any(w in rating for w in ["true","correct","accurate","verified","mostly true"]):
                    verdict = "REAL"
                else:
                    verdict = "UNCERTAIN"

                results.append({
                    "verdict":     verdict,
                    "raw_rating":  review.get("textualRating", ""),
                    "publisher":   review.get("publisher", {}).get("name", ""),
                    "url":         review.get("url", ""),
                    "claim_text":  item.get("text", ""),
                })

        if not results:
            return {"found": False, "reason": "No reviews parsed"}

        # Majority vote across all fact check results
        fake_count = sum(1 for r in results if r["verdict"] == "FAKE")
        real_count = sum(1 for r in results if r["verdict"] == "REAL")

        if fake_count > real_count:
            final_verdict = "FAKE"
        elif real_count > fake_count:
            final_verdict = "REAL"
        else:
            final_verdict = "UNCERTAIN"

        return {
            "found":         True,
            "verdict":       final_verdict,
            "total_checked": len(results),
            "fake_count":    fake_count,
            "real_count":    real_count,
            "top_result":    results[0],
            "all_results":   results,
        }

    except requests.exceptions.Timeout:
        return {"found": False, "reason": "Fact check API timed out"}
    except Exception as e:
        return {"found": False, "reason": str(e)}


def search_mongodb(headline: str) -> list:
    """
    Searches ALL news in MongoDB — no date restriction.
    Uses word overlap with stop word removal.
    Searches in batches to avoid RAM issues on Render free tier.
    """
    try:
        matches   = []
        batch_size = 300
        skip       = 0

        while True:
            batch = list(
                news_collection.find(
                    {},
                    {"title": 1, "description": 1, "source": 1, "verified_by_admin": 1, "createdAt": 1}
                )
                .sort("createdAt", -1)
                .skip(skip)
                .limit(batch_size)
            )

            if not batch:
                break

            for doc in batch:
                combined = doc.get("title", "") + " " + doc.get("description", "")
                score    = word_overlap_score(headline, combined)

                if score >= 0.25:
                    matches.append({"doc": doc, "score": score})

            # Stop scanning if we already have strong matches
            # and have checked at least 600 docs
            if skip >= 600 and len(matches) >= 10:
                break

            skip += batch_size

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:15]

    except Exception as e:
        print("MongoDB search error:", e)
        return []


@news_router.post("/verify-news")
async def verify_news(headline: str = Form(...)):
    headline = headline.strip()
    if not headline:
        raise HTTPException(status_code=400, detail="Headline is required.")

    start_time    = datetime.utcnow()
    user_polarity = get_polarity(headline)

    # ── Step 1: Search MongoDB (ALL time, no date filter) ──────────
    matches       = search_mongodb(headline)
    supporting    = []
    contradicting = []
    admin_verdict = None

    for m in matches:
        doc          = m["doc"]
        score        = m["score"]
        article_text = doc.get("title", "") + " " + doc.get("description", "")
        art_polarity = get_polarity(article_text)
        is_admin     = doc.get("verified_by_admin", False) or doc.get("source") == "Admin"
        weight       = score * (2.0 if is_admin else 1.0)

        entry = {
            "title":      doc.get("title", "")[:120],
            "source":     doc.get("source", ""),
            "similarity": round(score, 3),
            "polarity":   art_polarity,
            "is_admin":   is_admin,
        }

        # Admin article — strongest signal, set immediately
        if is_admin and admin_verdict is None:
            if art_polarity == "neutral" or art_polarity == user_polarity:
                admin_verdict = ("REAL", entry)
            else:
                admin_verdict = ("FAKE", entry)
            continue

        if art_polarity == user_polarity:
            supporting.append({"weight": weight, **entry})
        elif art_polarity != "neutral":
            contradicting.append({"weight": weight, **entry})

    # ── Step 2: Google Fact Check (ALL time, no date filter) ───────
    fact_check = check_google_factcheck(headline)

    # ── Step 3: Compute final verdict ─────────────────────────────
    prediction = "UNCERTAIN"
    confidence = 0.5
    reason     = "No strong evidence found."

    # Priority 1 — Admin record in your DB
    if admin_verdict:
        prediction = admin_verdict[0]
        confidence = 1.0
        reason     = f"{'Confirmed' if prediction == 'REAL' else 'Contradicted'} by admin-verified record."

    # Priority 2 — Google Fact Check has a clear verdict
    elif fact_check.get("found") and fact_check["verdict"] in ("REAL", "FAKE"):
        prediction = fact_check["verdict"]
        # Scale confidence based on how many fact checks agree
        total   = fact_check["total_checked"]
        agreeing = fact_check["fake_count"] if prediction == "FAKE" else fact_check["real_count"]
        confidence = round(0.75 + (agreeing / total) * 0.20, 4)  # 0.75–0.95
        reason  = (
            f"{agreeing}/{total} fact-check source(s) rate this as "
            f"'{fact_check['top_result']['raw_rating']}' "
            f"— {fact_check['top_result']['publisher']}."
        )

    # Priority 3 — Weighted DB voting
    elif matches:
        support_w = sum(e["weight"] for e in supporting)
        contra_w  = sum(e["weight"] for e in contradicting)
        total_w   = support_w + contra_w

        if total_w > 0:
            raw        = (support_w - contra_w) / total_w
            normalized = (raw + 1.0) / 2.0

            # Reduce confidence proportional to contradictions
            if contradicting:
                penalty    = (contra_w / total_w) * 0.4
                normalized = max(0.0, normalized - penalty)

            confidence = round(normalized, 4)

            if confidence >= 0.65:
                prediction = "REAL"
                reason     = (
                    f"{len(supporting)} article(s) support this claim, "
                    f"{len(contradicting)} contradict it."
                )
            elif confidence <= 0.35:
                prediction = "FAKE"
                reason     = (
                    f"{len(contradicting)} article(s) contradict this claim, "
                    f"only {len(supporting)} support it."
                )
            else:
                prediction = "UNCERTAIN"
                reason     = (
                    f"Mixed signals — {len(supporting)} supporting, "
                    f"{len(contradicting)} contradicting."
                )
        else:
            # Articles found but all neutral polarity
            prediction = "UNCERTAIN"
            confidence = 0.5
            reason     = "Related articles found but no clear positive/negative signal."

    # Priority 4 — Nothing found anywhere
    else:
        prediction = "UNCERTAIN"
        confidence = 0.0
        reason     = "No related articles found in database or fact-check records."

    response_time = (datetime.utcnow() - start_time).total_seconds()

    return {
        "status":                  "success",
        "headline":                headline,
        "prediction":              prediction,
        "confidence":              confidence,
        "reason":                  reason,
        "user_claim_polarity":     user_polarity,
        "fact_check":              fact_check,
        "supporting_articles":     supporting[:5],
        "contradicting_articles":  contradicting[:5],
        "total_db_matched":        len(matches),
        "response_time_seconds":   round(response_time, 3),
    }
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
app.include_router(news_router)
