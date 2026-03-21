# ============================================================
#  main.py  —  Fake News Verification API
# ============================================================

# ── Standard library ─────────────────────────────────────────
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime

# ── Third-party ───────────────────────────────────────────────
import joblib
import requests
from fastapi import FastAPI, APIRouter, Form, HTTPException
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

# ════════════════════════════════════════════════════════════
#  Global shared state (populated in background thread)
# ════════════════════════════════════════════════════════════
model      = None   # ML model (reserved for future use)
vectorizer = None   # TF-IDF vectorizer
news_docs  = []     # All documents pulled from MongoDB
tfidf_db   = None   # TF-IDF matrix of news_docs
collection = None   # MongoDB collection handle
_resources_ready = threading.Event()   # signals when loading is done


# ════════════════════════════════════════════════════════════
#  Resource loader  (runs in a background thread)
# ════════════════════════════════════════════════════════════
def load_resources_background():
    global model, vectorizer, news_docs, tfidf_db, collection

    try:
        print("🔹 Loading ML model …")
        model = joblib.load("fake_news_model.pkl")

        print("🔹 Loading TF-IDF vectorizer …")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")

        print("🔹 Connecting to MongoDB …")
        mongo_url = os.getenv("MONGO_URI")
        if not mongo_url:
            raise ValueError("MONGO_URI environment variable is not set.")

        client     = MongoClient(mongo_url, serverSelectionTimeoutMS=10_000)
        db         = client["fake_new_app"]
        collection = db["news"]

        news_docs = list(
            collection.find(
                {},
                {"title": 1, "description": 1, "source": 1,
                 "verified_by_admin": 1, "createdAt": 1}
            )
        )
        print(f"✅ Loaded {len(news_docs)} news documents from MongoDB.")

        if news_docs:
            texts    = [
                n.get("title", "") + " " + n.get("description", "")
                for n in news_docs
            ]
            tfidf_db = vectorizer.transform(texts)
            print("✅ TF-IDF matrix built.")

        _resources_ready.set()
        print("✅ Background loading complete.")

    except Exception as exc:
        print(f"❌ Background loading error: {exc}")
        # Still set the event so requests aren't blocked forever;
        # individual endpoints will detect None values and return 503.
        _resources_ready.set()


# ════════════════════════════════════════════════════════════
#  Lifespan  (replaces deprecated @app.on_event)
# ════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting — launching background resource loader …")
    thread = threading.Thread(target=load_resources_background, daemon=True)
    thread.start()
    yield                  # application runs here
    print("🛑 Server shutting down.")


# ════════════════════════════════════════════════════════════
#  App & router
# ════════════════════════════════════════════════════════════
app        = FastAPI(title="Fake News Verification API", lifespan=lifespan)
news_router = APIRouter(prefix="/news", tags=["News"])


# ════════════════════════════════════════════════════════════
#  Health endpoints
# ════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {
        "status":          "ok",
        "resources_ready": _resources_ready.is_set(),
    }


@app.head("/health")
def health_check_head():
    # HEAD must return no body — FastAPI strips it automatically,
    # but the function itself should return nothing.
    return None


# ════════════════════════════════════════════════════════════
#  NLP helpers
# ════════════════════════════════════════════════════════════
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
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "and", "or", "with", "this", "that", "it",
    "by", "from", "has", "have", "had", "be", "been", "being", "will",
    "would", "could", "should", "may", "might", "do", "does", "did",
    "not", "no", "its", "our", "their", "we", "he", "she", "they",
    "i", "you", "me", "him", "her", "us", "them", "as", "but", "if",
    "than", "so", "yet", "both", "either", "each", "too", "very", "just",
}


def get_polarity(text: str) -> str:
    """Returns 'positive', 'negative', or 'neutral' based on keyword matching."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    neg   = len(words & NEGATIVE_WORDS)
    pos   = len(words & POSITIVE_WORDS)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def word_overlap_score(a: str, b: str) -> float:
    """Jaccard-style overlap score with stop-word removal."""
    a_words = set(re.findall(r'\b\w+\b', a.lower())) - STOP_WORDS
    b_words = set(re.findall(r'\b\w+\b', b.lower())) - STOP_WORDS
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(len(a_words), len(b_words))


def find_similar_news_tfidf(user_text: str):
    """
    Returns (best_matching_doc, cosine_score) using the TF-IDF matrix.
    Returns (None, 0) when the matrix is not yet available.
    """
    if tfidf_db is None or not news_docs:
        return None, 0.0

    tfidf_user = vectorizer.transform([user_text])
    similarity = cosine_similarity(tfidf_user, tfidf_db)
    index      = int(similarity.argmax())
    score      = float(similarity[0][index])
    return news_docs[index], score


# ════════════════════════════════════════════════════════════
#  MongoDB search
# ════════════════════════════════════════════════════════════
def search_mongodb(headline: str) -> list:
    """
    Scans MongoDB in batches and returns articles with word-overlap ≥ 0.25,
    sorted by overlap score descending.  Stops early once we have 10+
    strong matches after scanning at least 600 docs.
    """
    if collection is None:
        print("⚠️  MongoDB collection is not initialised yet.")
        return []

    try:
        matches    = []
        batch_size = 300
        skip       = 0

        while True:
            batch = list(
                collection.find(
                    {},
                    {"title": 1, "description": 1, "source": 1,
                     "verified_by_admin": 1, "createdAt": 1}
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

            # Early stop once we have plenty of strong matches
            if skip >= 600 and len(matches) >= 10:
                break

            skip += batch_size

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:15]

    except Exception as exc:
        print(f"MongoDB search error: {exc}")
        return []


# ════════════════════════════════════════════════════════════
#  Google Fact Check API
# ════════════════════════════════════════════════════════════
def check_google_factcheck(claim: str) -> dict:
    """
    Queries the Google Fact Check Tools API and returns a structured result
    with a majority-vote verdict across the top-5 matching fact-check records.
    """
    api_key = os.getenv("GOOGLE_FACT_CHECK_KEY")
    if not api_key:
        return {"found": False, "reason": "GOOGLE_FACT_CHECK_KEY env var not set."}

    try:
        resp = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={
                "query":        claim,
                "key":          api_key,
                "languageCode": "en",
                "pageSize":     5,
            },
            timeout=6,
        )
        data = resp.json()

        if "claims" not in data or not data["claims"]:
            return {"found": False, "reason": "No fact-check records found."}

        results = []
        for item in data["claims"][:5]:
            for review in item.get("claimReview", []):
                rating = review.get("textualRating", "").lower()

                if any(w in rating for w in
                       ["false", "fake", "misleading", "incorrect",
                        "wrong", "pants on fire", "inaccurate"]):
                    verdict = "FAKE"
                elif any(w in rating for w in
                         ["true", "correct", "accurate", "verified",
                          "mostly true"]):
                    verdict = "REAL"
                else:
                    verdict = "UNCERTAIN"

                results.append({
                    "verdict":    verdict,
                    "raw_rating": review.get("textualRating", ""),
                    "publisher":  review.get("publisher", {}).get("name", ""),
                    "url":        review.get("url", ""),
                    "claim_text": item.get("text", ""),
                })

        if not results:
            return {"found": False, "reason": "No reviews could be parsed."}

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
        return {"found": False, "reason": "Fact check API timed out."}
    except Exception as exc:
        return {"found": False, "reason": str(exc)}


# ════════════════════════════════════════════════════════════
#  Verdict logic
# ════════════════════════════════════════════════════════════
def _classify_db_matches(headline: str, matches: list):
    """
    Splits DB matches into supporting / contradicting lists.
    An article is supporting if its topic entities overlap with the headline
    AND its polarity matches.  This avoids matching unrelated negative news
    against a negative headline.

    Returns: (supporting, contradicting, admin_verdict)
      admin_verdict is None or ("REAL"|"FAKE", entry_dict)
    """
    user_polarity  = get_polarity(headline)
    headline_words = set(re.findall(r'\b\w+\b', headline.lower())) - STOP_WORDS

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

        # Topic-entity check: at least 30 % key-word overlap so we don't
        # match unrelated news that happens to share polarity.
        art_words    = set(re.findall(r'\b\w+\b', article_text.lower())) - STOP_WORDS
        topic_overlap = (
            len(headline_words & art_words) / max(len(headline_words), 1)
        )
        if topic_overlap < 0.30 and not is_admin:
            continue   # skip weakly related articles

        entry = {
            "title":      doc.get("title", "")[:120],
            "source":     doc.get("source", ""),
            "similarity": round(score, 3),
            "polarity":   art_polarity,
            "is_admin":   is_admin,
        }

        # Admin article — highest-priority signal
        if is_admin and admin_verdict is None:
            # A neutral admin article is considered supporting (confirmed real)
            if art_polarity == "neutral" or art_polarity == user_polarity:
                admin_verdict = ("REAL", entry)
            else:
                admin_verdict = ("FAKE", entry)
            continue

        if art_polarity == user_polarity:
            supporting.append({"weight": weight, **entry})
        elif art_polarity != "neutral":
            contradicting.append({"weight": weight, **entry})

    return supporting, contradicting, admin_verdict


def _compute_verdict(
    admin_verdict, fact_check, matches, supporting, contradicting
):
    """
    Applies priority logic and returns (prediction, confidence, reason).
    Priority:  Admin record > Google Fact Check > Weighted DB vote > Unknown
    """
    # Priority 1 — Admin record
    if admin_verdict:
        verdict    = admin_verdict[0]
        confidence = 1.0
        reason     = (
            f"{'Confirmed' if verdict == 'REAL' else 'Contradicted'} "
            "by an admin-verified record."
        )
        return verdict, confidence, reason

    # Priority 2 — Google Fact Check
    if fact_check.get("found") and fact_check["verdict"] in ("REAL", "FAKE"):
        verdict   = fact_check["verdict"]
        total     = fact_check["total_checked"]
        agreeing  = (
            fact_check["fake_count"] if verdict == "FAKE"
            else fact_check["real_count"]
        )
        confidence = round(0.75 + (agreeing / total) * 0.20, 4)
        top        = fact_check["top_result"]
        reason     = (
            f"{agreeing}/{total} fact-check source(s) rate this as "
            f"'{top['raw_rating']}' — {top['publisher']}."
        )
        return verdict, confidence, reason

    # Priority 3 — Weighted DB vote
    if matches:
        support_w = sum(e["weight"] for e in supporting)
        contra_w  = sum(e["weight"] for e in contradicting)
        total_w   = support_w + contra_w

        if total_w == 0:
            return (
                "UNCERTAIN", 0.5,
                "Related articles found but all carry neutral polarity — "
                "no clear signal."
            )

        raw        = (support_w - contra_w) / total_w
        normalized = (raw + 1.0) / 2.0

        if contradicting:
            penalty    = (contra_w / total_w) * 0.4
            normalized = max(0.0, normalized - penalty)

        confidence = round(normalized, 4)

        if confidence >= 0.65:
            return (
                "REAL", confidence,
                f"{len(supporting)} article(s) support this claim, "
                f"{len(contradicting)} contradict it."
            )
        if confidence <= 0.35:
            return (
                "FAKE", confidence,
                f"{len(contradicting)} article(s) contradict this claim, "
                f"only {len(supporting)} support it."
            )
        return (
            "UNCERTAIN", confidence,
            f"Mixed signals — {len(supporting)} supporting, "
            f"{len(contradicting)} contradicting."
        )

    # Priority 4 — Nothing found
    return (
        "UNCERTAIN", 0.0,
        "No related articles found in the database or fact-check records."
    )


# ════════════════════════════════════════════════════════════
#  Main endpoint
# ════════════════════════════════════════════════════════════
@news_router.post("/verify-news")
async def verify_news(headline: str = Form(...)):
    headline = headline.strip()
    if not headline:
        raise HTTPException(status_code=400, detail="Headline is required.")

    # Return 503 while resources are still loading
    if not _resources_ready.is_set():
        raise HTTPException(
            status_code=503,
            detail="Server is still starting up. Please try again in a few seconds."
        )

    # If loading finished but MongoDB is still None, something went wrong
    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection is unavailable. Please try again later."
        )

    start_time = datetime.utcnow()

    # Step 1 — DB search
    matches                             = search_mongodb(headline)
    supporting, contradicting, admin_v  = _classify_db_matches(headline, matches)

    # Step 2 — Google Fact Check (runs in parallel would be ideal; kept sync for simplicity)
    fact_check = check_google_factcheck(headline)

    # Step 3 — Verdict
    prediction, confidence, reason = _compute_verdict(
        admin_v, fact_check, matches, supporting, contradicting
    )

    response_time = (datetime.utcnow() - start_time).total_seconds()

    return {
        "status":                 "success",
        "headline":               headline,
        "prediction":             prediction,
        "confidence":             confidence,
        "reason":                 reason,
        "user_claim_polarity":    get_polarity(headline),
        "fact_check":             fact_check,
        "supporting_articles":    supporting[:5],
        "contradicting_articles": contradicting[:5],
        "total_db_matched":       len(matches),
        "response_time_seconds":  round(response_time, 3),
    }


# ════════════════════════════════════════════════════════════
#  Router registration & entry-point
# ════════════════════════════════════════════════════════════
app.include_router(news_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
