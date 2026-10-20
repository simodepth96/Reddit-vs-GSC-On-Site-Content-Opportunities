import hashlib
import io
import json
import os
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")
SEED = 42

KG_URL = "https://kgsearch.googleapis.com/v1/entities:search"
JEV_URL = "https://api.typesafe.ai/v1/systemone"

st.set_page_config(page_title="Reddit vs GSC Opportunities", layout="wide")

OPP_COLS = ["Reddit Cluster", "GSC Cluster", "Confirmed Entity", "q_clicks",
            "ranking_queries", "community_angle", "content_opportunity_score"]

# Jev Noul questions: each returns P(yes) in [0, 1]. Answered in one call against one shared state.
JEV_QUESTIONS = {
    "brand_fit": {
        "type": "noul",
        "instructions": (
            "Is publisher_brand a credible, natural publisher of content on this topic, given its "
            "products, destinations, audience and USPs, and is the topic NOT excluded by its "
            "out_of_scope list?"),
        "criteria": {
            "true": "The topic sits squarely within what the brand sells, serves or is known for.",
            "false": "The topic is unrelated to the brand, off-brand, or listed as out of scope.",
        },
    },
    "entity_match": {
        "type": "noul",
        "instructions": (
            "Treat each confirmed_entity as verified ground truth from Google's Knowledge Graph. Do "
            "community_topic and search_topic concern the same entity, or closely related entities, "
            "so that one piece of content can serve both? A topic with no confirmed entity is "
            "unvalidated: do not assume it matches."),
        "criteria": {
            "true": "Both topics are about the same confirmed entity, or one is a closely related part of the other.",
            "false": "The confirmed entities differ materially, are missing, or the match rests only on loose wording.",
        },
    },
    "content_value": {
        "type": "noul",
        "instructions": (
            "Would new content from publisher_brand on this topic answer real, unmet community "
            "questions AND capture search demand the brand can realistically win?"),
        "criteria": {
            "true": "The community shows genuine unanswered questions and the search topic has meaningful demand.",
            "false": "Little unmet need, little demand, or the demand is already well served.",
        },
    },
}


# ---------------------------------------------------------------- helpers
def trunc(s, n=220):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def clean_text(s):
    s = (str(s).replace("\u2019", "'").replace("\u2018", "'")
         .replace("\u201c", '"').replace("\u201d", '"'))
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[>*_`#~\[\]]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def secret(name: str) -> str:
    """Env var first, then .streamlit/secrets.toml; empty string if neither is set."""
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        return str(st.secrets[name])
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def api_cache() -> dict:
    """Process-wide cache for Gemini / Knowledge Graph / Jev responses (thread-safe enough for dict get/set)."""
    return {}


@st.cache_data(show_spinner=False)
def load_table(name: str, data: bytes) -> pd.DataFrame:
    buf = io.BytesIO(data)
    return pd.read_csv(buf) if name.lower().endswith(".csv") else pd.read_excel(buf)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, max_len: int):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    m.max_seq_length = max_len
    return m


@st.cache_data(show_spinner=False)
def embed(model_name: str, max_len: int, texts: tuple) -> np.ndarray:
    model = load_model(model_name, max_len)
    return model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)


def cluster_embeddings(emb, cfg):
    """HDBSCAN on L2-normalised embeddings (euclidean == cosine ranking)."""
    from sklearn.cluster import HDBSCAN
    X = emb
    if cfg.get("use_umap") and len(emb) > 3:
        import umap
        nn = min(cfg.get("umap_neighbors", 15), len(emb) - 1)
        X = umap.UMAP(
            n_components=min(cfg.get("umap_dims", 5), max(2, len(emb) - 2)),
            n_neighbors=nn, min_dist=0.0, metric="cosine", random_state=SEED,
        ).fit_transform(emb)
    labels = HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
        cluster_selection_method=cfg["method"],
        metric="euclidean",
    ).fit_predict(X)
    if (labels != -1).sum() == 0:
        st.warning("HDBSCAN found no clusters - lower min_cluster_size/min_samples. "
                   "Everything was put in one cluster so the run can continue.")
        labels = np.zeros(len(emb), dtype=int)
    return labels


def centroids(emb, labels):
    ids = sorted(l for l in set(labels) if l != -1)
    C = np.vstack([emb[labels == i].mean(0) for i in ids])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return ids, C


def absorb_noise(emb, labels, min_sim):
    if min_sim is None or not (labels == -1).any():
        return labels
    ids, C = centroids(emb, labels)
    noise = np.where(labels == -1)[0]
    sims = emb[noise] @ C.T
    ok = sims.max(1) >= min_sim
    labels = labels.copy()
    labels[noise[ok]] = np.array(ids)[sims.argmax(1)[ok]]
    return labels


def tfidf_labels(df, n_terms=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = df[df.cluster != -1].groupby("cluster")["text"].apply(" ".join)
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(docs)
    except ValueError:
        return {cid: f"cluster {cid}" for cid in docs.index}
    terms = np.array(vec.get_feature_names_out())
    return {cid: ", ".join(terms[X[i].toarray().ravel().argsort()[::-1][:n_terms]])
            for i, cid in enumerate(docs.index)}


def find_typos(queries, allow, max_freq):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    toks = [re.findall(r"[a-z0-9']+", q_) for q_ in queries]
    freq = Counter(t for ts in toks for t in set(ts))
    bad = {}
    for t, f in freq.items():
        if f > max_freq or t in allow or any(c.isdigit() for c in t) or len(t) < 3:
            continue
        if spell.unknown([t]):
            fix = spell.correction(t)
            if fix and fix != t:
                bad[t] = fix
    return bad


# ------------------------------------------------------------ pre-processing
def prep_queries(df, typo_mode, allow, max_freq):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    df = df[df["text"] != ""]
    df = df[df["text"].map(str.isascii)]
    df = df.groupby("text", as_index=False)["weight"].sum()

    if typo_mode != "off":
        typos = find_typos(df["text"].tolist(), allow, max_freq)
        if typos:
            pat = re.compile(r"(?<![a-z0-9'])(" + "|".join(map(re.escape, typos)) + r")(?![a-z0-9'])")
            has = df["text"].str.contains(pat)
            if typo_mode == "drop":
                df = df[~has]
            else:
                df.loc[has, "text"] = df.loc[has, "text"].str.replace(
                    pat, lambda m: typos[m.group(1)], regex=True)
                df = df.groupby("text", as_index=False)["weight"].sum()
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def prep_reddit(df, brands, brand_mode, min_words, min_chars, min_alpha, drop_regex):
    df = df.copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0).clip(lower=0)
    df["text"] = df["text"].map(clean_text)
    df = df[~df["text"].str.lower().isin(["[deleted]", "[removed]", "deleted", "removed"])]

    if brands:
        pat = re.compile(r"\b(" + "|".join(re.escape(b) for b in brands) + r")(?:'s)?\b", re.I)
        if brand_mode == "drop_rows":
            df = df[~df["text"].str.contains(pat)]
        else:
            df["text"] = (df["text"].str.replace(pat, " ", regex=True)
                          .str.replace(r"\s+", " ", regex=True).str.strip())

    if drop_regex:
        df = df[~df["text"].str.lower().str.contains(drop_regex, regex=True)]

    df = df[df["text"].str.split().str.len() >= min_words]
    df = df[df["text"].str.len() >= min_chars]
    alpha = df["text"].map(lambda t: sum(c.isalpha() for c in t) / max(1, len(t.replace(" ", ""))))
    df = df[alpha >= min_alpha]

    return (df.assign(_k=df["text"].str.lower()).sort_values("weight", ascending=False)
            .drop_duplicates("_k").drop(columns="_k").reset_index(drop=True))


# --------------------------------------------------------------- API clients
def run_parallel(fn, items, workers):
    """Run fn over items in threads. Returns (results aligned to items with None on failure, error strings)."""
    results, errors = [None] * len(items), []
    if not items:
        return results, errors
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            try:
                results[futs[f]] = f.result()
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
    return results, errors


def gemini_name_clusters(api_key, model, kind, items, cache):
    """items: [{id, examples, key_terms?}] -> {id: {name, entity}}. Chunked, cached."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    out = {}
    for i in range(0, len(items), 40):
        chunk = items[i:i + 40]
        prompt = (
            f"You label clusters of {kind} for an SEO team. For each cluster return:\n"
            "- name: a natural-language topic name, 3-8 words, phrased the way a person would "
            "describe or search the topic (no jargon, no quotation marks).\n"
            "- entity: the single most specific real-world entity (place, organisation, product, "
            "vessel, event, person or concept) the cluster is about that Google's Knowledge Graph "
            "is likely to know, using its canonical name. Use an empty string if the cluster is "
            "generic and has no such entity.\n"
            'Return ONLY a JSON array like [{"id": 0, "name": "...", "entity": "..."}] that '
            "covers every id.\n\nClusters:\n" + json.dumps(chunk, ensure_ascii=False)
        )
        ck = hashlib.sha1((model + prompt).encode()).hexdigest()
        data = cache.get(ck)
        if data is None:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   temperature=0.2))
            txt = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
            data = json.loads(txt)
            cache[ck] = data
        for d in data:
            out[int(d["id"])] = dict(name=str(d.get("name", "")).strip(),
                                     entity=str(d.get("entity", "")).strip())
    return out


def _similar(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return False
    return a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.75


def kg_lookup(entity, key, cache):
    """Google Knowledge Graph Search: first result whose name matches the entity, else None."""
    ck = "kg:" + entity.lower()
    if ck in cache:
        return cache[ck]
    r = requests.get(KG_URL, params={"query": entity, "key": key, "limit": 5, "languages": "en"},
                     timeout=30)
    r.raise_for_status()
    hit = None
    for el in r.json().get("itemListElement", []):
        res = el.get("result", {})
        if _similar(entity, res.get("name", "")):
            hit = dict(
                name=res["name"], kg_id=res.get("@id", ""),
                types=[t for t in res.get("@type", []) if t != "Thing"],
                description=res.get("description", ""),
                summary=trunc(res.get("detailedDescription", {}).get("articleBody", ""), 300),
            )
            break
    cache[ck] = hit
    return hit


def ent_label(hit):
    if not hit:
        return ""
    t = ", ".join(hit["types"][:2])
    return f"{hit['name']} ({t})" if t else hit["name"]


def jev_ask(state, questions, model, key, cache):
    """One TypeSafe Jev System One call: one state, several Noul questions -> answers dict."""
    payload = {"model": model, "state": state, "questions": questions}
    ck = "jev:" + hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if ck in cache:
        return cache[ck]
    r = None
    for attempt in range(4):
        r = requests.post(JEV_URL, json=payload, timeout=60,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code in (429, 500, 502, 503, 504, 529):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        ans = r.json()["answers"]
        cache[ck] = ans
        return ans
    r.raise_for_status()


# ----------------------------------------------------------------- analysis
def analyse(q, r, Eq, Er, P, keys, cache):
    warns = []
    q, r = q.copy(), r.copy()
    q["cluster"] = absorb_noise(Eq, cluster_embeddings(Eq, P["q_cfg"]), P["q_cfg"]["noise_absorb"])
    r["cluster"] = absorb_noise(Er, cluster_embeddings(Er, P["r_cfg"]), P["r_cfg"]["noise_absorb"])
    q_ids, Cq = centroids(Eq, q["cluster"].values)
    r_ids, Cr = centroids(Er, r["cluster"].values)

    # ---- cluster summaries (keyword labels are fallbacks until Gemini names them)
    q_rows = []
    for cid, g in q[q.cluster != -1].groupby("cluster"):
        g = g.sort_values("weight", ascending=False)
        q_rows.append(dict(q_cluster=int(cid), fallback=g.iloc[0]["text"], n_queries=len(g),
                           clicks=int(g["weight"].sum()), top_queries=" | ".join(g["text"].head(6)),
                           examples=g["text"].head(8).tolist(), key_terms=""))
    q_sum = pd.DataFrame(q_rows).sort_values("clicks", ascending=False).reset_index(drop=True)

    terms = tfidf_labels(r)
    r_rows = []
    for i, cid in enumerate(r_ids):
        m = np.where(r.cluster.values == cid)[0]
        order = m[np.argsort(-(Er[m] @ Cr[i]))]
        g = r.iloc[m]
        r_rows.append(dict(r_cluster=int(cid), fallback=terms.get(cid, f"cluster {cid}"),
                           n_comments=len(g), total_score=int(g["weight"].sum()),
                           signal=len(g) + np.log1p(g["weight"]).sum(),
                           representative_comment=trunc(r.iloc[order[0]]["text"]),
                           examples=[trunc(r.iloc[k]["text"], 200) for k in order[:3]],
                           key_terms=terms.get(cid, "")))
    r_sum = pd.DataFrame(r_rows).sort_values("signal", ascending=False).reset_index(drop=True)

    # ---- Gemini: natural-language names + candidate Google entity
    st.write("Naming clusters with Gemini…")

    def name_set(kind, df, id_col):
        items = [dict(id=int(c), key_terms=kt, examples=ex)
                 for c, kt, ex in zip(df[id_col], df["key_terms"], df["examples"])]
        try:
            return gemini_name_clusters(keys["gemini"], P["gemini_model"], kind, items, cache)
        except Exception as e:
            warns.append(f"Gemini naming failed for {kind} ({type(e).__name__}: {e}); "
                         "keyword labels used instead.")
            return {}

    qn = name_set("Google Search Console query clusters", q_sum, "q_cluster")
    rn = name_set("Reddit comment clusters", r_sum, "r_cluster")
    q_sum["GSC Cluster"] = [qn.get(c, {}).get("name") or fb for c, fb in zip(q_sum.q_cluster, q_sum.fallback)]
    r_sum["Reddit Cluster"] = [rn.get(c, {}).get("name") or fb for c, fb in zip(r_sum.r_cluster, r_sum.fallback)]
    q_sum["entity_guess"] = [qn.get(c, {}).get("entity", "") for c in q_sum.q_cluster]
    r_sum["entity_guess"] = [rn.get(c, {}).get("entity", "") for c in r_sum.r_cluster]

    # ---- Knowledge Graph: confirm the entities
    st.write("Confirming entities in the Knowledge Graph…")
    ents = sorted({e for e in list(q_sum.entity_guess) + list(r_sum.entity_guess) if e})
    hits, errs = run_parallel(lambda e: kg_lookup(e, keys["kg"], cache), ents, P["workers"])
    if errs:
        warns.append(f"Knowledge Graph lookups failed for {len(errs)} of {len(ents)} entities ({errs[0]}).")
    kg = dict(zip(ents, hits))
    for df in (q_sum, r_sum):
        df["kg"] = [kg.get(e) for e in df["entity_guess"]]
        df["Confirmed Entity"] = [ent_label(h) for h in df["kg"]]
    kg_confirmed = int((q_sum["Confirmed Entity"] != "").sum() + (r_sum["Confirmed Entity"] != "").sum())
    kg_total = len(q_sum) + len(r_sum)

    # ---- match Reddit clusters to query clusters (cosine is internal only)
    S = Cq @ Cr.T
    qd, rd = q_sum.set_index("q_cluster"), r_sum.set_index("r_cluster")
    pairs = []
    for j, rid in enumerate(r_ids):
        for i in np.argsort(-S[:, j])[:P["top_q_per_r"]]:
            if S[i, j] >= P["match_threshold"]:
                pairs.append(dict(rid=int(rid), qid=int(q_ids[i]), sim=float(S[i, j])))
    pairs = sorted(pairs, key=lambda p: -p["sim"])[:P["max_pairs"]]

    # ---- Jev: content_opportunity_score per matched pair
    def entity_obj(hit):
        if not hit:
            return "No confirmed Google Knowledge Graph entity"
        return dict(name=hit["name"], kg_id=hit["kg_id"], types=hit["types"],
                    description=hit["description"], summary=hit["summary"])

    brand = dict(name=P["brand_name"], products_destinations_audience_usps=P["brand_specs"],
                 out_of_scope=P["brand_oos"] or "none specified")
    states = []
    for p in pairs:
        rr, qr = rd.loc[p["rid"]], qd.loc[p["qid"]]
        states.append(dict(
            publisher_brand=brand,
            community_topic=dict(name=rr["Reddit Cluster"], confirmed_entity=entity_obj(rr["kg"]),
                                 comments=int(rr["n_comments"]), total_upvotes=int(rr["total_score"]),
                                 example_comments=rr["examples"]),
            search_topic=dict(name=qr["GSC Cluster"], confirmed_entity=entity_obj(qr["kg"]),
                              queries=int(qr["n_queries"]), clicks=int(qr["clicks"]),
                              top_queries=qr["examples"][:6]),
        ))

    scores = []
    if states:
        st.write(f"Scoring {len(states)} matched pairs with Jev…")
        answers, errs = run_parallel(
            lambda s: jev_ask(s, JEV_QUESTIONS, P["jev_model"], keys["jev"], cache),
            states, P["workers"])
        if errs and all(a is None for a in answers):
            raise RuntimeError(f"TypeSafe Jev failed: {errs[0]}")
        if errs:
            warns.append(f"Jev failed for {len(errs)} of {len(states)} pairs ({errs[0]}); "
                         "their score is empty.")
        w = P["score_weights"]
        for a in answers:
            scores.append(np.nan if a is None else round(
                w[0] * a["brand_fit"]["noul"] + w[1] * a["entity_match"]["noul"]
                + w[2] * a["content_value"]["noul"], 3))

    rows = []
    for p, sc in zip(pairs, scores):
        rr, qr = rd.loc[p["rid"]], qd.loc[p["qid"]]
        labels = []
        for lab in (rr["Confirmed Entity"], qr["Confirmed Entity"]):
            if lab and lab not in labels:
                labels.append(lab)
        rows.append({"Reddit Cluster": rr["Reddit Cluster"], "GSC Cluster": qr["GSC Cluster"],
                     "Confirmed Entity": " / ".join(labels), "q_clicks": int(qr["clicks"]),
                     "ranking_queries": qr["top_queries"], "community_angle": rr["representative_comment"],
                     "content_opportunity_score": sc})
    opps = (pd.DataFrame(rows, columns=OPP_COLS)
            .sort_values("content_opportunity_score", ascending=False, na_position="last")
            .reset_index(drop=True))

    # ---- gaps (both directions)
    r_gap = []
    for j, rid in enumerate(r_ids):
        if S[:, j].max() < P["match_threshold"]:
            rr = rd.loc[rid]
            r_gap.append({"Reddit Cluster": rr["Reddit Cluster"], "Confirmed Entity": rr["Confirmed Entity"],
                          "n_comments": int(rr["n_comments"]),
                          "representative_comment": rr["representative_comment"],
                          "Closest GSC Cluster": qd.loc[q_ids[S[:, j].argmax()], "GSC Cluster"]})
    r_gaps = pd.DataFrame(r_gap, columns=["Reddit Cluster", "Confirmed Entity", "n_comments",
                                          "representative_comment", "Closest GSC Cluster"])
    r_gaps = r_gaps.sort_values("n_comments", ascending=False).reset_index(drop=True)

    q_gap = []
    for i, qid in enumerate(q_ids):
        if S[i].max() < P["match_threshold"]:
            qr = qd.loc[qid]
            q_gap.append({"GSC Cluster": qr["GSC Cluster"], "Confirmed Entity": qr["Confirmed Entity"],
                          "n_queries": int(qr["n_queries"]), "clicks": int(qr["clicks"]),
                          "top_queries": qr["top_queries"],
                          "Closest Reddit Cluster": rd.loc[r_ids[S[i].argmax()], "Reddit Cluster"]})
    q_gaps = pd.DataFrame(q_gap, columns=["GSC Cluster", "Confirmed Entity", "n_queries", "clicks",
                                          "top_queries", "Closest Reddit Cluster"])
    q_gaps = q_gaps.sort_values("clicks", ascending=False).reset_index(drop=True)

    return dict(
        opps=opps, r_gaps=r_gaps, q_gaps=q_gaps,
        q_sum=q_sum[["GSC Cluster", "Confirmed Entity", "n_queries", "clicks", "top_queries"]],
        r_sum=r_sum[["Reddit Cluster", "Confirmed Entity", "n_comments", "representative_comment"]],
        kg_confirmed=kg_confirmed, kg_total=kg_total, warnings=warns,
        brand=P["brand_name"])


def to_excel(res):
    buf = io.BytesIO()
    sheets = {"opportunities": res["opps"], "reddit_gaps": res["r_gaps"], "query_gaps": res["q_gaps"],
              "query_clusters": res["q_sum"], "reddit_clusters": res["r_sum"]}
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
st.title("Content Gaps Explorer - Reddit vs Google Search Console")

with st.sidebar:
    st.header("Inputs")
    q_file = st.file_uploader("GSC queries (xlsx/csv)", type=["xlsx", "xls", "csv"])
    r_file = st.file_uploader("Reddit comments (xlsx/csv)", type=["xlsx", "xls", "csv"])

    def pick(df, label, default, key):
        cols = list(df.columns)
        return st.selectbox(label, cols, index=cols.index(default) if default in cols else 0, key=key)

    q_df = r_df = None
    if q_file:
        q_df = load_table(q_file.name, q_file.getvalue())
        QUERY_COL = pick(q_df, "Query column", "Query", "qc")
        CLICKS_COL = pick(q_df, "Clicks column", "Clicks", "cc")
    if r_file:
        r_df = load_table(r_file.name, r_file.getvalue())
        REDDIT_COL = pick(r_df, "Comment column", "keyword", "rc")
        SCORE_COL = pick(r_df, "Score column", "score", "sc")

    with st.expander("API keys", expanded=True):
        GEMINI_KEY = st.text_input("Gemini API key", secret("GEMINI_API_KEY"), type="password")
        GEMINI_MODEL = st.text_input("Gemini model", "gemini-flash-latest")
        KG_KEY = st.text_input("Knowledge Graph Search API key", secret("GOOGLE_KG_API_KEY"), type="password")
        TS_KEY = st.text_input("TypeSafe API key", secret("TYPESAFE_API_KEY"), type="password")
        JEV_MODEL = st.text_input("Jev model", "jev-latest")
        WORKERS = st.number_input("Parallel API calls", 1, 32, 8)

    with st.expander("Brand (content publisher)", expanded=True):
        BRAND_NAME = st.text_input("Brand name", "")
        BRAND_SPECS = st.text_area("Brand specs: products, destinations, audience, USPs", "")
        BRAND_OOS = st.text_area("Out of scope topics (optional)", "")
        st.caption("Content opportunity score weights (normalised to sum to 1)")
        w1 = st.slider("Brand fit", 0.0, 1.0, 0.4, 0.05)
        w2 = st.slider("Entity match (Knowledge Graph)", 0.0, 1.0, 0.3, 0.05)
        w3 = st.slider("Content value (demand + unmet need)", 0.0, 1.0, 0.3, 0.05)

    with st.expander("Embedding"):
        MODEL_NAME = st.selectbox("Model", ["all-mpnet-base-v2", "all-MiniLM-L6-v2"])
        MAX_SEQ_LEN = st.number_input("Max sequence length", 32, 512, 256, step=32)

    with st.expander("Query pre-processing"):
        TYPO_MODE = st.selectbox("Typo handling", ["drop", "correct", "off"])
        TYPO_MAX_FREQ = st.number_input("Typo candidate max token frequency", 1, 20, 1)
        ALLOW = st.text_input("Allowlist (comma-separated)", "marella, douro, tui")

    with st.expander("Reddit pre-processing"):
        BRANDS = st.text_input("Brands to remove from Reddit data (comma-separated)", "")
        BRAND_MODE = st.selectbox("Brand mode", ["drop_rows", "strip_term"])
        MIN_WORDS = st.number_input("Min words", 0, 200, 8)
        MIN_CHARS = st.number_input("Min chars", 0, 1000, 40)
        MIN_ALPHA = st.slider("Min alpha ratio", 0.0, 1.0, 0.6, 0.05)
        DROP_REGEX = st.text_input(
            "Drop regex (blank = off)",
            r"^(thanks|thank you|thx|great|sounds (good|great)|congrats)\b")

    with st.expander("HDBSCAN: queries"):
        q_mcs = st.number_input("min_cluster_size", 2, 100, 2, key="q_mcs")
        q_ms = st.number_input("min_samples", 1, 100, 1, key="q_ms")
        q_method = st.selectbox("method", ["leaf", "eom"], key="q_method")
        q_umap = st.checkbox("Use UMAP", False, key="q_umap")
        q_umap_dims = st.number_input("UMAP dims", 2, 50, 5, key="q_ud")
        q_umap_nn = st.number_input("UMAP neighbors", 2, 100, 15, key="q_un")
        q_absorb_on = st.checkbox("Absorb noise", True, key="q_abs_on")
        q_absorb = st.slider("Noise absorb cosine", 0.0, 1.0, 0.55, 0.01, key="q_abs")

    with st.expander("HDBSCAN: Reddit"):
        r_mcs = st.number_input("min_cluster_size", 2, 100, 5, key="r_mcs")
        r_ms = st.number_input("min_samples", 1, 100, 2, key="r_ms")
        r_method = st.selectbox("method", ["leaf", "eom"], key="r_method")
        r_umap = st.checkbox("Use UMAP", True, key="r_umap")
        r_umap_dims = st.number_input("UMAP dims", 2, 50, 5, key="r_ud")
        r_umap_nn = st.number_input("UMAP neighbors", 2, 100, 15, key="r_un")
        r_absorb_on = st.checkbox("Absorb noise", False, key="r_abs_on")
        r_absorb = st.slider("Noise absorb cosine", 0.0, 1.0, 0.55, 0.01, key="r_abs")

    with st.expander("Matching"):
        MATCH_T = st.slider("Match threshold (centroid cosine)", 0.0, 1.0, 0.35, 0.01)
        TOP_Q = st.number_input("Query clusters per Reddit cluster", 1, 10, 2)
        MAX_PAIRS = st.number_input("Max pairs to score with Jev", 1, 1000, 200)

    missing = [n for n, ok in [
        ("both files", bool(q_file and r_file)), ("brand name", bool(BRAND_NAME.strip())),
        ("brand specs", bool(BRAND_SPECS.strip())), ("Gemini key", bool(GEMINI_KEY)),
        ("Knowledge Graph key", bool(KG_KEY)), ("TypeSafe key", bool(TS_KEY))] if not ok]
    run = st.button("Run analysis", type="primary", disabled=bool(missing))
    if missing:
        st.caption("Needed: " + ", ".join(missing))

# ---------------------------------------------------------------------- run
if run:
    try:
        wsum = (w1 + w2 + w3) or 1.0
        P = dict(
            q_cfg=dict(min_cluster_size=int(q_mcs), min_samples=int(q_ms), method=q_method,
                       use_umap=q_umap, umap_dims=int(q_umap_dims), umap_neighbors=int(q_umap_nn),
                       noise_absorb=q_absorb if q_absorb_on else None),
            r_cfg=dict(min_cluster_size=int(r_mcs), min_samples=int(r_ms), method=r_method,
                       use_umap=r_umap, umap_dims=int(r_umap_dims), umap_neighbors=int(r_umap_nn),
                       noise_absorb=r_absorb if r_absorb_on else None),
            match_threshold=MATCH_T, top_q_per_r=int(TOP_Q), max_pairs=int(MAX_PAIRS),
            workers=int(WORKERS), gemini_model=GEMINI_MODEL.strip(), jev_model=JEV_MODEL.strip(),
            brand_name=BRAND_NAME.strip(), brand_specs=BRAND_SPECS.strip(), brand_oos=BRAND_OOS.strip(),
            score_weights=(w1 / wsum, w2 / wsum, w3 / wsum),
        )
        keys = dict(gemini=GEMINI_KEY.strip(), kg=KG_KEY.strip(), jev=TS_KEY.strip())
        cache = api_cache()

        with st.status("Running analysis…", expanded=True) as status:
            st.write("Pre-processing…")
            q = q_df[[QUERY_COL, CLICKS_COL]].dropna(subset=[QUERY_COL]).copy()
            q.columns = ["text", "weight"]
            allow = {w.strip().lower() for w in ALLOW.split(",") if w.strip()}
            q = prep_queries(q, TYPO_MODE, allow, int(TYPO_MAX_FREQ))

            r = r_df[[REDDIT_COL, SCORE_COL]].dropna(subset=[REDDIT_COL]).copy()
            r.columns = ["text", "weight"]
            brands = [b.strip() for b in BRANDS.split(",") if b.strip()]
            r = prep_reddit(r, brands, BRAND_MODE, int(MIN_WORDS), int(MIN_CHARS),
                            MIN_ALPHA, DROP_REGEX.strip() or None)
            if len(q) < 3 or len(r) < 3:
                raise ValueError(f"Too few rows after cleaning (queries={len(q)}, comments={len(r)}).")

            st.write(f"Embedding {len(q)} queries and {len(r)} comments (first run downloads the model)…")
            Eq = embed(MODEL_NAME, int(MAX_SEQ_LEN), tuple(q["text"]))
            Er = embed(MODEL_NAME, int(MAX_SEQ_LEN), tuple(r["text"]))

            st.write("Clustering and matching…")
            res = analyse(q, r, Eq, Er, P, keys, cache)
            res.update(n_q=len(q), n_r=len(r))
            st.session_state["res"] = res
            status.update(label="Done", state="complete", expanded=False)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------ results
res = st.session_state.get("res")
if not res:
    st.info("Upload both files, add the brand and API keys in the sidebar, then click **Run analysis**.")
    st.stop()

for w in res["warnings"]:
    st.warning(w)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Queries (clean)", res["n_q"])
c2.metric("Reddit comments (clean)", res["n_r"])
c3.metric("Opportunities scored", len(res["opps"]))
c4.metric("Clusters with confirmed entity", f"{res['kg_confirmed']}/{res['kg_total']}")

st.download_button(
    "Download results (xlsx)", to_excel(res), "content_opportunities.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

tabs = st.tabs(["Opportunities", "Reddit gaps", "Query gaps", "Query clusters", "Reddit clusters"])

with tabs[0]:
    if res["opps"].empty:
        st.warning("No matches above the threshold. Lower the match threshold in the sidebar.")
    else:
        st.caption(f"content_opportunity_score (0–1) is scored for {res['brand']}: a weighted blend of "
                   "Jev's brand-fit, entity-match and content-value probabilities, validated against "
                   "Knowledge Graph entities.")
        st.dataframe(res["opps"], width="stretch", hide_index=True)

with tabs[1]:
    st.caption("Community topics with no ranking-query match: new content / keyword ideas.")
    st.dataframe(res["r_gaps"], width="stretch", hide_index=True)

with tabs[2]:
    st.caption("Ranking queries with no community match.")
    st.dataframe(res["q_gaps"], width="stretch", hide_index=True)

with tabs[3]:
    st.dataframe(res["q_sum"], width="stretch", hide_index=True)

with tabs[4]:
    st.dataframe(res["r_sum"], width="stretch", hide_index=True)
