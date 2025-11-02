# ingest.py
import sys

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from app import create_app, db, Staff, Event, StaffEvent, Topic, EventTopic, StaffTopic, thai_seg, get_embedder, \
    EMBED_DIM, embed_passage
import numpy as np

app = create_app()


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
    doc_info  = doc_info.reset_index(drop=True)
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
        SELECT se.staff_id, et.topic_id, SUM(COALESCE(et.probability,0.0)) AS strength
        FROM staff_event se
        JOIN event_topic et ON et.event_id = se.event_id
        GROUP BY se.staff_id, et.topic_id
        ON CONFLICT (staff_id, topic_id) DO UPDATE
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
    store = defaultdict(float)   # holds the aggregated probability
    seen_last = {}               # for 'last'

    for r in rows:
        eid = r["event_id"]
        tid = r["topic_id"]
        p   = r.get("probability", 0.0)
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
                et_rows.append({"event_id": event.event_id, "topic_id": topic.topic_id, "probability": r.Probability})
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


if __name__ == "__main__":
    # Example placeholders — replace with your real objects/frames
    excel_file = sys.argv[1]
    events_df = pd.read_excel(excel_file)  # must include: event_id,title,date,text,staff_ids
    run_ingest(events_df)
    # test_upsert()
