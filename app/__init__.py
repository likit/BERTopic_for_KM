# app.py
import os
import re
import sys

import click
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from sqlalchemy import text, Index, UniqueConstraint, Computed
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from sqlalchemy.orm import relationship
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

    def __str__(self):
        return self.name


class StaffEvent(db.Model):
    __tablename__ = "staff_event"
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.staff_id"), primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("events.event_id"), primary_key=True)
    event = relationship("Event", backref=db.backref('staff_events'))
    staff = relationship(Staff, backref=db.backref('staff_events'))


class Event(db.Model):
    __tablename__ = "events"
    event_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title    = db.Column(db.String)
    date     = db.Column(db.Date)


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

    data = []
    for evt_id, title, top_keywords in rows:
        dict_ = {"event_id": evt_id, "title": title, "keywords": top_keywords}
        event = Event.query.get(evt_id)
        dict_['staff'] = [s.staff for s in event.staff_events]
        data.append(dict_)

    return data
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
            topics = search_topics(query, topk=5)
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


app = create_app()


@app.cli.command('ingest_data')
@click.argument('infile')
def ingest_data(infile):
    import pandas as pd
    from sqlalchemy import text
    from sqlalchemy.dialects.postgresql import insert
    import numpy as np

    def upsert_bulk(model, rows, pk_fields):
        if not rows: return
        tbl = model.__table__
        stmt = insert(tbl).values(rows)
        update_cols = {c.name: c for c in tbl.columns if c.name not in pk_fields}
        stmt = stmt.on_conflict_do_update(index_elements=pk_fields, set_=update_cols)
        db.session.execute(stmt)

    def build_topic_rows(topic_info):
        embedder = get_embedder()
        out = []
        for r in topic_info.itertuples(index=False):
            tid = int(r.Topic)
            if tid == -1:  # skip outlier cluster
                continue
            label = r.Name
            seg = thai_seg(f"{label} {r.Top_n_words.replace(' - ', ' ')}")
            # emb = embedder.encode([label], normalize_embeddings=True)[0]
            emb = embed_passage(embedder, seg)  # not plain encode
            if EMBED_DIM != len(emb):
                raise ValueError(f"EMBED_DIM={EMBED_DIM} but embedder returns {len(emb)} dims—adjust in app.py")
            out.append({
                "topic_id": tid, "label": label, "top_keywords": r.Top_n_words,
                "seg_text": seg, "label_embedding": np.asarray(emb, dtype=np.float32)
            })
        return out

    def wire_event_topics(events_df, doc_info):
        events_df = events_df.reset_index(drop=True)
        doc_info = doc_info.reset_index(drop=True)
        rows = []
        for i, di in doc_info.iterrows():
            tid = int(di["Topic"])
            if tid == -1: continue
            evt_id = events_df.loc[i, "event_id"]
            prob = float(di.get("Probability", 1.0) or 1.0)
            rows.append({"event_id": evt_id, "topic_id": tid, "probability": prob})
        return rows

    def refresh_staff_topic():
        db.session.execute(text("""
                                INSERT INTO staff_topic(staff_id, topic_id, strength)
                                SELECT se.staff_id, et.topic_id, SUM(COALESCE(et.probability, 0.0)) AS strength
                                FROM staff_event se
                                         JOIN event_topic et ON et.event_id = se.event_id
                                GROUP BY se.staff_id, et.topic_id ON CONFLICT (staff_id, topic_id) DO
                                UPDATE
                                    SET strength = EXCLUDED.strength
                                """))

    REQUIRED = ["topic_id", "label", "top_keywords", "seg_text", "label_embedding"]

    def sanitize_topic_rows(topic_rows, embed_dim):
        cleaned = []
        for i, r in enumerate(topic_rows):
            # required keys
            miss = [k for k in REQUIRED if k not in r]
            if miss:
                raise ValueError(f"topic_rows[{i}] missing keys: {miss}")

            # types / coercions
            r["topic_id"] = int(r["topic_id"])
            r["label"] = "" if r["label"] is None else str(r["label"])
            r["top_keywords"] = "" if r["top_keywords"] is None else str(r["top_keywords"])
            r["seg_text"] = "" if r["seg_text"] is None else str(r["seg_text"])

            # embedding checks
            emb = np.asarray(r["label_embedding"], dtype=np.float32)
            if emb.ndim != 1 or emb.shape[0] != embed_dim:
                raise ValueError(f"topic_rows[{i}] embedding shape {emb.shape} != ({embed_dim},)")
            if not np.isfinite(emb).all():
                raise ValueError(f"topic_rows[{i}] embedding has NaN/Inf")
            r["label_embedding"] = emb

            cleaned.append(r)
        return cleaned

    from collections import defaultdict
    import math

    def dedup_event_topic_rows(rows, agg="max"):
        """
        Deduplicate rows by (event_id, topic_id).
        agg: 'max' | 'sum' | 'last' — how to combine duplicate probabilities.
        """
        store = defaultdict(float)  # holds the aggregated probability
        seen_last = {}  # for 'last'

        for r in rows:
            eid = r["event_id"]
            tid = r["topic_id"]
            p = r.get("probability", 0.0)
            if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
                p = 0.0

            key = (eid, tid)
            if agg == "sum":
                store[key] += float(p)
            elif agg == "max":
                store[key] = max(store[key], float(p))
            elif agg == "last":
                seen_last[key] = float(p)
            else:
                raise ValueError("agg must be 'max' | 'sum' | 'last'")

        if agg == "last":
            return [{"event_id": k[0], "topic_id": k[1], "probability": v}
                    for k, v in seen_last.items()]
        else:
            return [{"event_id": k[0], "topic_id": k[1], "probability": v}
                    for k, v in store.items()]

    def run_ingest(events_df: pd.DataFrame):
        print('ingesting..')
        with app.app_context():
            # 1) staff
            print('\tupserting staff..')
            upsert_bulk(Staff, [{"name": r.name}
                                for r in events_df.itertuples(index=False)
                                ], pk_fields=["staff_id"])

            # 2) events
            print('\tupserting events..')
            upsert_bulk(Event, [{"title": r.title}
                                for r in events_df.itertuples(index=False)
                                ], pk_fields=["event_id"])
            db.session.commit()

            # 3) staff_event
            print('\tupserting staff + events..')
            titles = list(set(events_df["title"].dropna().tolist()))
            names = list(set(events_df["name"].dropna().tolist()))

            event_map = dict(db.session.query(Event.title, Event.event_id)
                             .filter(Event.title.in_(titles)).all())
            staff_map = dict(db.session.query(Staff.name, Staff.staff_id)
                             .filter(Staff.name.in_(names)).all())

            # Report missing (don’t break on the first)
            missing_events = [t for t in titles if t not in event_map]
            missing_staffs = [n for n in names if n not in staff_map]
            if missing_events or missing_staffs:
                print(f"Missing events: {len(missing_events)}; Missing staff: {len(missing_staffs)}")
                # optionally log a few samples

            # Build rows, skipping unknowns; also dedupe
            pair_set = set()
            for r in events_df.itertuples(index=False):
                eid = event_map.get(r.title)
                sid = staff_map.get(r.name)
                if eid and sid:
                    pair_set.add((sid, eid))

            se_rows = [{"staff_id": sid, "event_id": eid} for (sid, eid) in pair_set]
            db.session.execute(
                insert(StaffEvent).values(se_rows).on_conflict_do_nothing(
                    index_elements=["staff_id", "event_id"]
                )
            )
            db.session.commit()

            # se_rows = []
            # for r in events_df.itertuples(index=False):
            #     event = Event.query.filter_by(title=r.title).first()
            #     staff = Staff.query.filter_by(name=r.name).first()
            #     if not event or not staff:
            #         print(f'{r.title}\n{r.name}')
            #         break
            #     else:
            #         se_rows += [{"staff_id": staff.staff_id, "event_id": event.event_id}]
            # if se_rows:
            #     upsert_bulk(StaffEvent, se_rows, pk_fields=["staff_id", "event_id"])

            # 4) topics
            print('\tupserting topics..')
            topic_rows = build_topic_rows(events_df)
            topic_rows = sanitize_topic_rows(topic_rows, EMBED_DIM)
            by_id = {}
            for r in topic_rows:
                by_id[r["topic_id"]] = r
            topic_rows = list(by_id.values())

            print("rows:", len(topic_rows), "unique topic_ids:", len({r["topic_id"] for r in topic_rows}))

            def chunked(seq, n=2000):
                for i in range(0, len(seq), n):
                    yield seq[i:i + n]

            total = 0
            for chunk in chunked(topic_rows, 2000):
                total += upsert_bulk(Topic, chunk, pk_fields=["topic_id"]) or 0
                db.session.commit()

            # 5) event_topic
            print('\tupserting events + topics..')
            et_rows = []
            for r in events_df.itertuples(index=False):
                event = Event.query.filter_by(title=r.title).first()
                topic = Topic.query.filter_by(label=r.Name).first()
                if topic:
                    et_rows.append(
                        {"event_id": event.event_id, "topic_id": topic.topic_id, "probability": r.Probability})
            et_rows = dedup_event_topic_rows(et_rows, agg="max")  # or 'sum' if you want to accumulate
            upsert_bulk(EventTopic, et_rows, pk_fields=["event_id", "topic_id"])
            db.session.commit()

            # 6) staff_topic (pre-aggregate)
            print('\trefreshing staff topic..')
            refresh_staff_topic()
            db.session.commit()

    def test_upsert():
        with app.app_context():
            import numpy as np
            test_rows = [{
                "topic_id": 999999,
                "label": "Test topic",
                "top_keywords": "foo:0.5 bar:0.4",
                "seg_text": "Test topic foo bar",
                "label_embedding": np.zeros(EMBED_DIM, dtype=np.float32),
            }]

            print("rows to upsert:", test_rows[0].keys())
            n = upsert_bulk(Topic, test_rows, pk_fields=["topic_id"])
            db.session.commit()
            print("upserted:", n)

            # Verify:
            rows = db.session.execute(db.text(
                "SELECT topic_id, label, top_keywords, label_embedding::text FROM topics WHERE topic_id=999999"
            )).all()
            print(rows)

    # Example placeholders — replace with your real objects/frames
    excel_file = infile
    events_df = pd.read_excel(excel_file)  # must include: event_id,title,date,text,staff_ids
    run_ingest(events_df)


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
