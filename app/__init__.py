# app.py
import os
import re

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from sqlalchemy import text, Index, UniqueConstraint, Computed
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from wtforms import StringField
from wtforms.validators import DataRequired
from sentence_transformers import SentenceTransformer
from pythainlp import word_tokenize


load_dotenv()


db = SQLAlchemy()
EMBED_DIM = 384  # if you choose multilingual-e5-base

def embed_query(embedder, q: str):
    return embedder.encode([q], normalize_embeddings=True)[0]

def embed_passage(embedder, s: str):
    return embedder.encode([s], normalize_embeddings=True)[0]

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")

def thai_seg(s: str) -> str:
    return "" if not s else " ".join(word_tokenize(s, engine="newmm"))

def maybe_segment_query(q: str) -> str:
    return thai_seg(q) if THAI_RE.search(q) else q

def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URL"]
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]
    db.init_app(app)

    with app.app_context():
        # Ensure required extensions exist
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
        db.session.commit()
        db.create_all()
        # Build ivfflat after data (safe to run repeatedly)
        db.session.execute(text("""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname='ix_topics_label_embedding_ivfflat'
              ) THEN
                CREATE INDEX ix_topics_label_embedding_ivfflat
                ON topics USING ivfflat (label_embedding vector_cosine_ops)
                WITH (lists = 100);
              END IF;
            END $$;"""))
        db.session.commit()

    register_routes(app)
    return app

# ---------- MODELS ----------

class Staff(db.Model):
    __tablename__ = "staff"
    staff_id   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name       = db.Column(db.String)


class Event(db.Model):
    __tablename__ = "events"
    event_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title    = db.Column(db.String)
    date     = db.Column(db.Date)


class StaffEvent(db.Model):
    __tablename__ = "staff_event"
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.staff_id"), primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("events.event_id"), primary_key=True)


class Topic(db.Model):
    __tablename__ = "topics"
    topic_id      = db.Column(db.Integer, primary_key=True)
    label         = db.Column(db.Text, nullable=False)
    top_keywords  = db.Column(db.Text)
    # Thai-friendly searchable text you’ll store (segmented); see ingest code
    seg_text      = db.Column(db.Text, nullable=True)
    # Generated tsvector from seg_text for FTS
    topic_tsv     = db.Column(TSVECTOR)
    # Semantic vector
    label_embedding = db.Column(Vector(EMBED_DIM))

    __table_args__ = (
        # FTS index
        Index("ix_topics_topic_tsv_gin", topic_tsv, postgresql_using="gin"),
        # Trigram indexes
        Index("ix_topics_label_trgm", "label", postgresql_using="gin",
              postgresql_ops={"label": "gin_trgm_ops"}),
        Index("ix_topics_kw_trgm", "top_keywords", postgresql_using="gin",
              postgresql_ops={"top_keywords": "gin_trgm_ops"}),
        Index(
            "ix_topics_segtext_trgm",
            "seg_text",
            postgresql_using="gin",
            postgresql_ops={"seg_text": "gin_trgm_ops"},
        ),
        # (Exact) vector index is optional—ivfflat created in create_app()
    )


class EventTopic(db.Model):
    __tablename__ = "event_topic"
    event_id    = db.Column(db.Integer, db.ForeignKey("events.event_id"), primary_key=True)
    topic_id    = db.Column(db.Integer, db.ForeignKey("topics.topic_id"), primary_key=True)
    probability = db.Column(db.Float)


class StaffTopic(db.Model):
    __tablename__ = "staff_topic"
    staff_id  = db.Column(db.Integer, db.ForeignKey("staff.staff_id"), primary_key=True)
    topic_id  = db.Column(db.Integer, db.ForeignKey("topics.topic_id"), primary_key=True)
    strength  = db.Column(db.Float)

# ---------- HYBRID SEARCH HELPERS ----------

def get_embedder():
    # Reuse your BERTopic embedder if you pass it in; else default miniLM
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def thai_seg(s: str) -> str:
    if not s: return ""
    return " ".join(word_tokenize(s, engine="newmm"))

def vec_literal(v):
    # pgvector text literal: [0.1,0.2,...]
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

W_SEM, W_FTS, W_TRGM = 0.6, 0.3, 0.2

def search_topics(query: str, topk: int = 10, embedder=None, min_score=0.05):
    if embedder is None:
        embedder = get_embedder()
    qv = embed_query(embedder, query)
    qvec = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"

    seg_q = maybe_segment_query(query)
    # qv = embedder.encode([query], normalize_embeddings=True)[0]
    # qvec = vec_literal(qv)

    # 1) semantic
    sem = db.session.execute(text("""
      SELECT topic_id,
             COALESCE(NULLIF(label,''), 'Topic '||topic_id::text) AS label,
             (1 - (label_embedding <=> :qvec)) AS sem_score
      FROM topics
      WHERE label_embedding IS NOT NULL
      ORDER BY label_embedding <=> :qvec
      LIMIT :k
    """), {"qvec": qvec, "k": topk}).mappings().all()

    # 2) FTS (segmented)
    fts = db.session.execute(text("""
      SELECT topic_id,
             COALESCE(NULLIF(label,''), 'Topic '||topic_id::text) AS label,
             ts_rank_cd(topic_tsv, websearch_to_tsquery('simple', unaccent(:q))) AS fts_score
      FROM topics
      WHERE topic_tsv @@ websearch_to_tsquery('simple', unaccent(:q))
      ORDER BY fts_score DESC
      LIMIT :k
    """), {"q": seg_q, "k": topk}).mappings().all()

    # 3) trigram (include seg_text too)
    db.session.execute(text("SET LOCAL pg_trgm.similarity_threshold = 0.1"))
    trgm = db.session.execute(text("""
      SELECT topic_id,
             COALESCE(NULLIF(label,''), 'Topic '||topic_id::text) AS label,
             GREATEST(
               similarity(unaccent(label),       unaccent(:q)),
               similarity(unaccent(top_keywords),unaccent(:q)),
               similarity(unaccent(seg_text),    unaccent(:q))
             ) AS trgm_score
      FROM topics
      ORDER BY trgm_score DESC
      LIMIT :k
    """), {"q": seg_q, "k": topk}).mappings().all()

    # --- merge
    from collections import defaultdict
    comb = defaultdict(lambda: {"label":"", "sem":0.0, "fts":0.0, "trgm":0.0})

    for r in sem:
        comb[r["topic_id"]]["label"] = r["label"]
        comb[r["topic_id"]]["sem"]   = float(r["sem_score"])

    if fts:
        mx = max(float(r["fts_score"]) for r in fts) or 1.0
        for r in fts:
            comb[r["topic_id"]]["fts"] = float(r["fts_score"])/mx
            if not comb[r["topic_id"]]["label"]:
                comb[r["topic_id"]]["label"] = r["label"]

    if trgm:
        mx = max(float(r["trgm_score"]) for r in trgm) or 1.0
        for r in trgm:
            comb[r["topic_id"]]["trgm"] = float(r["trgm_score"])/mx
            if not comb[r["topic_id"]]["label"]:
                comb[r["topic_id"]]["label"] = r["label"]

    # compute final score
    W_SEM, W_FTS, W_TRGM = 0.6, 0.3, 0.1  # or your preferred weights
    scored = []
    for tid, d in comb.items():
        lbl = d["label"] or f"Topic {tid}"
        final = W_SEM*d["sem"] + W_FTS*d["fts"] + W_TRGM*d["trgm"]
        scored.append({"topic_id": tid, "label": lbl, "score": round(final,4)})

    filtered = [t for t in scored if t["score"] > min_score]
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:topk]
#
#     # 1) semantic via pgvector (cosine similarity)
#     sem = db.session.execute(text("""
#       SELECT topic_id, label, (1 - (label_embedding <=> :qvec)) AS sem_score
#       FROM topics
#       ORDER BY label_embedding <=> :qvec
#       LIMIT :k
#     """), {"qvec": qvec, "k": topk}).mappings().all()
#
#     # 2) FTS (websearch syntax)
#     fts = db.session.execute(text("""
#       SELECT topic_id, ts_rank_cd(topic_tsv, websearch_to_tsquery('simple', unaccent(:q))) AS fts_score
#       FROM topics
#       WHERE topic_tsv @@ websearch_to_tsquery('simple', unaccent(:q))
#       ORDER BY fts_score DESC
#       LIMIT :k
#     """), {"q": query, "k": topk}).mappings().all()
#
#     # 3) trigram (fallback for Thai phrases / substrings)
#     db.session.execute(text("SET LOCAL pg_trgm.similarity_threshold = 0.1"))  # looser for Thai
#     trgm = db.session.execute(text("""
#       SELECT topic_id,
#         GREATEST(similarity(unaccent(label), unaccent(:q)),
#                  similarity(unaccent(top_keywords), unaccent(:q))) AS trgm_score
#       FROM topics
#       WHERE unaccent(label) % unaccent(:q) OR unaccent(top_keywords) % unaccent(:q)
#       ORDER BY trgm_score DESC
#       LIMIT :k
#     """), {"q": query, "k": topk}).mappings().all()
#
#     # combine (normalize fts/trgm, keep sem as-is in [0,1])
#     from collections import defaultdict
#     comb = defaultdict(lambda: {"label":"", "sem":0, "fts":0, "trgm":0})
#     for r in sem: comb[r["topic_id"]].update(label=r["label"], sem=float(r["sem_score"]))
#     if fts:
#         mx = max(float(r["fts_score"]) for r in fts) or 1.0
#         for r in fts: comb[r["topic_id"]]["fts"] = float(r["fts_score"])/mx
#     if trgm:
#         mx = max(float(r["trgm_score"]) for r in trgm) or 1.0
#         for r in trgm: comb[r["topic_id"]]["trgm"] = float(r["trgm_score"])/mx
#
#     scored = []
#     for tid, d in comb.items():
#         final = W_SEM*d["sem"] + W_FTS*d["fts"] + W_TRGM*d["trgm"]
#         scored.append({"topic_id": tid, "label": d["label"], "score": round(final, 4)})
#     scored.sort(key=lambda x: x["score"], reverse=True)
#     return scored[:topk]

def staff_breakdown(topic_scores, topk=10):
    tids = [t["topic_id"] for t in topic_scores]
    weights = {t["topic_id"]: t["score"] for t in topic_scores}
    qmarks = ",".join(map(str, tids))

    rows = db.session.execute(db.text(f"""
      SELECT s.staff_id, s.name, st.topic_id, st.strength, t.label
      FROM staff_topic st
      JOIN staff s ON s.staff_id = st.staff_id
      JOIN topics t ON t.topic_id = st.topic_id
      WHERE st.topic_id IN ({qmarks})
    """)).all()

    from collections import defaultdict
    contrib = defaultdict(lambda: defaultdict(float))
    meta    = {}
    for sid, name, tid, strength, label in rows:
        meta[sid] = name
        contrib[sid][(tid, label)] += strength * weights[tid]

    out = []
    for sid, parts in contrib.items():
        items = sorted(parts.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        out.append({
            "staff_id": sid,
            "name": meta[sid],
            "score": round(sum(v for _, v in items), 3),
            "contributions": [
                {"topic_id": tid, "label": lbl, "score": round(v,3)}
                for (tid, lbl), v in items
            ]
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def staff_for_topics(topic_scores, topk=20):
    if not topic_scores: return []
    tids = [t["topic_id"] for t in topic_scores]
    weights = {t["topic_id"]: t["score"] for t in topic_scores}
    qmarks = ",".join(str(t) for t in tids)

    rows = db.session.execute(text(f"""
      SELECT st.staff_id, s.name, st.topic_id, st.strength
      FROM staff_topic st
      JOIN staff s ON s.staff_id = st.staff_id
      WHERE st.topic_id IN ({qmarks})
    """)).all()

    from collections import defaultdict
    agg, meta = defaultdict(float), {}
    for staff_id, name, tid, strength in rows:
        agg[staff_id] += float(strength) * float(weights.get(tid, 0.0))
        meta[staff_id] = name

    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [{"staff_id": sid, "name": meta[sid], "score": round(sc, 3)} for sid, sc in ranked]


def event_for_topics(topic_scores):
    if not topic_scores: return []
    tids = ",".join([str(t["topic_id"]) for t in topic_scores])

    rows = db.session.execute(text(f"""
      SELECT evt.event_id, evt.title, t.top_keywords FROM event_topic et
      JOIN topics t ON t.topic_id = et.topic_id
      JOIN events evt on et.event_id = evt.event_id
      WHERE t.topic_id IN ({tids})
    """)).all()

    return [{"event_id": eid, "title": title, "keywords": keywords} for eid, title, keywords in rows]
# ---------- ROUTES ----------

class SearchForm(FlaskForm):
    query = StringField('Query', validators=[DataRequired()])


def register_routes(app: Flask):
    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/search")
    def search_api():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "missing ?q="}), 400
        topics = search_topics(q, topk=3)
        staff = staff_for_topics(topics, topk=5)
        return jsonify({"query": q, "topics": topics, "staff": staff})

    @app.route("/search-form", methods=["GET", "POST"])
    def search():
        form = SearchForm()
        if request.method == "POST":
            query = form.query.data
            topics = search_topics(query, topk=3)
            if topics:
                staff = staff_for_topics(topics, topk=5)
                events = event_for_topics(topics)
                breakdown = staff_breakdown(topics)
            else:
                staff = []
                events = []
                breakdown = []
            return render_template('search_form.html',
                                   form=form, topics=topics, staff=staff, events=events,
                                   breakdown=breakdown)
        return render_template('search_form.html', form=form)


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
