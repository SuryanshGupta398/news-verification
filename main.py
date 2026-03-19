from fastapi import FastAPI
import joblib
from pymongo import MongoClient
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# from fastapi import HTTPException

# import numpy as np

# from sklearn.metrics.pairwise import cosine_similarity
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from datetime import datetime

# def utc_now_iso():
#     return datetime.utcnow().isoformat()

# @app.get("/health")
# def health_check():
#     return {"status": "ok", "time": utc_now_iso()}

# @app.head("/health")
# def health_check_head():
#     return {"status": "ok"}
    
model = None
vectorizer = None
news_docs = []
tfidf_db = None


@app.on_event("startup")
def load_resources():
    global model, vectorizer, news_docs, tfidf_db

    try:
        print("🔹 Loading model...")
        model = joblib.load("fake_news_model.pkl")

        print("🔹 Loading vectorizer...")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")

        print("🔹 Connecting MongoDB...")
        mongo_url = os.getenv("MONGO_URI")
        if not mongo_url:
            print("❌ MONGO_URI not found")
            return

        client = MongoClient(mongo_url)

        db = client["fake_new_app"]
        collection = db["news"]

        news_docs = list(collection.find({}, {"title":1,"description":1,"source":1}))

        print(f"✅ Loaded {len(news_docs)} news articles")

        if news_docs:
            texts = [
                n.get("title","") + " " + n.get("description","")
                for n in news_docs
            ]
            tfidf_db = vectorizer.transform(texts)

        print("✅ Startup complete")

    except Exception as e:
        print("❌ ERROR DURING STARTUP:", str(e))
        
# class NewsRequest(BaseModel):
#     text: str

# # -----------------------------
# # Find Similar News
# # -----------------------------
# def find_similar_news(user_text):
#     if len(news_docs) == 0:
#         return None, 0

#     tfidf_user = vectorizer.transform([user_text])

#     similarity = cosine_similarity(tfidf_user, tfidf_db)

#     index = similarity.argmax()
#     score = similarity[0][index]

#     return news_docs[index], score
# import re
# from sklearn.metrics.pairwise import cosine_similarity

# # ── Extract named entities (people, orgs) from text ───────────────
# def extract_entities(text: str) -> set:
#     """
#     Simple regex-based NER — finds capitalized word sequences.
#     Lightweight alternative to spaCy/NLTK.
#     """
#     text_clean = re.sub(r'[^\w\s]', ' ', text)
#     # Match sequences of capitalized words (names, orgs)
#     entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text_clean)
#     # Also grab common Hindi name patterns via known list
#     return {e.lower() for e in entities}


# # ── Sentiment polarity — no library needed ────────────────────────
# NEGATIVE_WORDS = {
#     "died", "dead", "death", "killed", "murdered", "lost", "defeated",
#     "arrested", "jailed", "fired", "resigned", "crashed", "failed",
#     "attacked", "bombed", "expelled", "suspended", "banned", "collapsed",
#     "injured", "hospitalized", "missing", "destroyed", "abolished",
#     "mart", "mara", "maut", "hatya", "nahi", "gaya", "gir",
# }

# POSITIVE_WORDS = {
#     "won", "win", "victory", "launched", "inaugurated", "appointed",
#     "elected", "promoted", "visited", "announced", "achieved", "grew",
#     "increased", "recovered", "released", "celebrated", "awarded",
#     "daura", "yatra", "bole", "khela", "jeeta", "aaya",
# }

# def get_polarity(text: str) -> str:
#     """Returns 'positive', 'negative', or 'neutral'"""
#     words = set(re.findall(r'\b\w+\b', text.lower()))
#     neg_count = len(words & NEGATIVE_WORDS)
#     pos_count = len(words & POSITIVE_WORDS)
#     if neg_count > pos_count:
#         return "negative"
#     if pos_count > neg_count:
#         return "positive"
#     return "neutral"

# def find_all_similar_news(user_text: str, top_k: int = 10, min_score: float = 0.25):
#     if not news_docs or tfidf_db is None:
#         return []

#     tfidf_user = vectorizer.transform([user_text])
#     scores     = cosine_similarity(tfidf_user, tfidf_db)[0]

#     results = [
#         (news_docs[i], float(scores[i]))
#         for i in range(len(news_docs))
#         if float(scores[i]) >= min_score
#     ]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return results[:top_k]


# def get_polarity(text: str) -> str:
#     words = set(re.findall(r'\b\w+\b', text.lower()))
#     neg   = len(words & NEGATIVE_WORDS)
#     pos   = len(words & POSITIVE_WORDS)
#     if neg > pos: return "negative"
#     if pos > neg: return "positive"
#     return "neutral"


# def compute_final_confidence(
#     user_polarity: str,
#     matches: list,
# ) -> dict:
#     """
#     For each matched article:
#       - If polarity AGREES   with user claim  → SUPPORTING  → adds to confidence
#       - If polarity OPPOSES  user claim       → CONTRADICTING → reduces confidence
#       - If neutral                            → ignored

#     Final confidence = (supporting_weight - contradicting_weight) / total_weight
#     Clamped to [0, 1].
#     """
#     supporting     = []   # articles that agree
#     contradicting  = []   # articles that oppose
#     neutral_docs   = []   # articles with no clear polarity
#     admin_verdict  = None # admin overrides everything

#     for doc, score in matches:
#         full_text      = doc.get("title", "") + " " + doc.get("description", "")
#         article_polarity = get_polarity(full_text)
#         source         = doc.get("source", "user")

#         entry = {
#             "title":      doc.get("title", "")[:100],
#             "source":     source,
#             "similarity": round(score, 4),
#             "polarity":   article_polarity,
#             "weight":     score * (2.0 if source == "admin" else 1.0),
#         }

#         # Admin record — immediately determine verdict
#         if source == "admin":
#             if article_polarity == user_polarity or article_polarity == "neutral":
#                 admin_verdict = ("REAL", entry)
#             else:
#                 admin_verdict = ("FAKE", entry)
#             continue

#         if article_polarity == "neutral":
#             neutral_docs.append(entry)
#         elif article_polarity == user_polarity:
#             supporting.append(entry)
#         else:
#             contradicting.append(entry)

#     # ── Admin verdict overrides everything ────────────────────────
#     if admin_verdict:
#         verdict, admin_entry = admin_verdict
#         return {
#             "prediction":            verdict,
#             "confidence":            1.0,
#             "reason":                f"{'Confirmed' if verdict == 'REAL' else 'Contradicted'} by admin-verified record.",
#             "supporting_articles":   supporting,
#             "contradicting_articles":contradicting,
#             "admin_article":         admin_entry,
#             "total_checked":         len(matches),
#         }

#     # ── Weighted vote ─────────────────────────────────────────────
#     support_weight = sum(e["weight"] for e in supporting)
#     contra_weight  = sum(e["weight"] for e in contradicting)
#     total_weight   = support_weight + contra_weight

#     if total_weight == 0:
#         return {
#             "prediction":            "UNCERTAIN",
#             "confidence":            0.0,
#             "reason":                "Related articles found but none have clear positive/negative polarity.",
#             "supporting_articles":   [],
#             "contradicting_articles":[],
#             "admin_article":         None,
#             "total_checked":         len(matches),
#         }

#     # Raw score = how much support outweighs contradiction
#     raw_confidence = (support_weight - contra_weight) / total_weight

#     # Normalize to [0, 1]
#     # +1.0 = fully supported, 0.0 = perfectly split, -1.0 = fully contradicted
#     normalized = (raw_confidence + 1.0) / 2.0

#     # Scale down if contradictions exist (penalty per contradicting article)
#     if contradicting:
#         penalty     = (contra_weight / total_weight) * 0.5
#         normalized  = max(0.0, normalized - penalty)

#     # Determine final verdict from normalized score
#     if normalized >= 0.65:
#         prediction = "REAL"
#         reason = (
#             f"{len(supporting)} article(s) support this claim, "
#             f"{len(contradicting)} oppose it. "
#             f"Confidence adjusted for contradictions."
#         )
#     elif normalized <= 0.35:
#         prediction = "FAKE"
#         reason = (
#             f"{len(contradicting)} article(s) contradict this claim, "
#             f"only {len(supporting)} support it."
#         )
#     else:
#         prediction = "UNCERTAIN"
#         reason = (
#             f"Mixed evidence — {len(supporting)} supporting, "
#             f"{len(contradicting)} contradicting articles found."
#         )

#     return {
#         "prediction":            prediction,
#         "confidence":            round(normalized, 4),
#         "reason":                reason,
#         "supporting_articles":   supporting,
#         "contradicting_articles":contradicting,
#         "admin_article":         None,
#         "total_checked":         len(matches),
#     }


# @app.post("/verify")
# def verify_news(data: NewsRequest):
#     text = data.text.strip()
#     if not text:
#         raise HTTPException(status_code=400, detail="Text cannot be empty.")

#     user_polarity = get_polarity(text)
#     matches       = find_all_similar_news(text, top_k=10, min_score=0.25)

#     if matches:
#         result = compute_final_confidence(user_polarity, matches)

#         best_doc, best_score = matches[0]
#         return {
#             **result,
#             "user_claim_polarity": user_polarity,
#             "best_match_title":    best_doc.get("title", ""),
#             "best_match_score":    round(best_score, 4),
#         }

#     # No DB matches at all → ML fallback
#     vec        = vectorizer.transform([text])
#     prediction = model.predict(vec)[0]
#     confidence = float(model.predict_proba(vec).max())

#     return {
#         "prediction":            "REAL" if prediction == 1 else "FAKE",
#         "confidence":            round(confidence, 4),
#         "reason":                "No DB match found — ML model used.",
#         "supporting_articles":   [],
#         "contradicting_articles":[],
#         "admin_article":         None,
#         "total_checked":         0,
#         "user_claim_polarity":   user_polarity,
#         "best_match_title":      None,
#         "best_match_score":      None,
#     }

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
