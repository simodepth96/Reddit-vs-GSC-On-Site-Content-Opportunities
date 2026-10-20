import io
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
SEED = 42

st.set_page_config(page_title="Reddit vs GSC Opportunities", layout="wide")


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
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(docs)
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
    log = []
    n0 = len(df)
    df = df.copy()
    df["text"] = df["text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    df = df[df["text"] != ""]

    bad = ~df["text"].map(str.isascii)
    log.append(f"Dropped {bad.sum()} queries with non-ASCII/accented characters")
    df = df[~bad]

    n1 = len(df)
    df = df.groupby("text", as_index=False)["weight"].sum()
    log.append(f"Merged {n1 - len(df)} duplicates (clicks summed)")

    if typo_mode != "off":
        typos = find_typos(df["text"].tolist(), allow, max_freq)
        if typos:
            log.append(f"Typo candidates: {typos}")
            pat = re.compile(r"(?<![a-z0-9'])(" + "|".join(map(re.escape, typos)) + r")(?![a-z0-9'])")
            has = df["text"].str.contains(pat)
            if typo_mode == "drop":
                log.append(f"Dropped {has.sum()} typo queries: {df.loc[has, 'text'].tolist()}")
                df = df[~has]
            else:
                df.loc[has, "text"] = df.loc[has, "text"].str.replace(
                    pat, lambda m: typos[m.group(1)], regex=True)
                df = df.groupby("text", as_index=False)["weight"].sum()
                log.append(f"Corrected {has.sum()} typo queries")
        else:
            log.append("No typos found")
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    log.append(f"Queries: {n0} → {len(df)}")
    return df, log


def prep_reddit(df, brands, brand_mode, min_words, min_chars, min_alpha, drop_regex):
    log = [("raw", len(df))]
    df = df.copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0).clip(lower=0)
    df["text"] = df["text"].map(clean_text)
    df = df[~df["text"].str.lower().isin(["[deleted]", "[removed]", "deleted", "removed"])]
    log.append(("after removing deleted/removed", len(df)))

    if brands:
        pat = re.compile(r"\b(" + "|".join(re.escape(b) for b in brands) + r")(?:'s)?\b", re.I)
        if brand_mode == "drop_rows":
            df = df[~df["text"].str.contains(pat)]
        else:
            df["text"] = (df["text"].str.replace(pat, " ", regex=True)
                          .str.replace(r"\s+", " ", regex=True).str.strip())
        log.append((f"after brand removal ({brand_mode}: {', '.join(brands)})", len(df)))

    if drop_regex:
        df = df[~df["text"].str.lower().str.contains(drop_regex, regex=True)]
        log.append(("after dropping pleasantries", len(df)))

    df = df[df["text"].str.split().str.len() >= min_words]
    log.append((f"after >= {min_words} words", len(df)))
    df = df[df["text"].str.len() >= min_chars]
    log.append((f"after >= {min_chars} chars", len(df)))
    alpha = df["text"].map(lambda t: sum(c.isalpha() for c in t) / max(1, len(t.replace(" ", ""))))
    df = df[alpha >= min_alpha]
    log.append((f"after alpha ratio >= {min_alpha}", len(df)))

    df = (df.assign(_k=df["text"].str.lower()).sort_values("weight", ascending=False)
          .drop_duplicates("_k").drop(columns="_k").reset_index(drop=True))
    log.append(("after de-duplication", len(df)))
    return df, pd.DataFrame(log, columns=["step", "rows"])


# ----------------------------------------------------------------- analysis
def analyse(q, r, Eq, Er, brand_emb, P):
    q, r = q.copy(), r.copy()
    q["cluster"] = absorb_noise(Eq, cluster_embeddings(Eq, P["q_cfg"]), P["q_cfg"]["noise_absorb"])
    r["cluster"] = absorb_noise(Er, cluster_embeddings(Er, P["r_cfg"]), P["r_cfg"]["noise_absorb"])

    stats = pd.DataFrame([
        dict(set=n, clusters=d.loc[d.cluster != -1, "cluster"].nunique(),
             noise_rows=int((d.cluster == -1).sum()), noise_pct=f"{(d.cluster == -1).mean():.0%}")
        for n, d in (("Query", q), ("Reddit", r))
    ])

    q_ids, Cq = centroids(Eq, q["cluster"].values)
    r_ids, Cr = centroids(Er, r["cluster"].values)

    # query cluster summary
    q_rows = []
    for cid, g in q[q.cluster != -1].groupby("cluster"):
        g = g.sort_values("weight", ascending=False)
        q_rows.append(dict(q_cluster=cid, label=g.iloc[0]["text"], n_queries=len(g),
                           clicks=int(g["weight"].sum()),
                           top_queries=" | ".join(g["text"].head(6))))
    q_sum = pd.DataFrame(q_rows).sort_values("clicks", ascending=False).reset_index(drop=True)

    # reddit cluster summary
    terms = tfidf_labels(r)
    r_rows = []
    for i, cid in enumerate(r_ids):
        m = np.where(r.cluster.values == cid)[0]
        rep = m[(Er[m] @ Cr[i]).argmax()]
        g = r.iloc[m]
        r_rows.append(dict(r_cluster=cid, label=terms[cid], n_comments=len(g),
                           total_score=int(g["weight"].sum()),
                           signal=len(g) + np.log1p(g["weight"]).sum(),
                           representative_comment=trunc(r.iloc[rep]["text"])))
    r_sum = pd.DataFrame(r_rows).sort_values("signal", ascending=False).reset_index(drop=True)

    # cluster-to-cluster similarity
    S = Cq @ Cr.T
    sim_stats = (pd.Series(S.ravel()).describe(percentiles=[.5, .75, .9, .95])
                 .round(3).to_frame("cosine").reset_index())

    qd = q_sum.set_index("q_cluster")
    rd = r_sum.set_index("r_cluster")
    q_demand = np.log1p(qd["clicks"])
    q_demand = q_demand / (q_demand.max() or 1)
    r_signal = np.log1p(rd["signal"])
    r_signal = r_signal / (r_signal.max() or 1)

    pairs = []
    for j, rid in enumerate(r_ids):
        for i in np.argsort(-S[:, j])[:P["top_q_per_r"]]:
            qid, sim = q_ids[i], float(S[i, j])
            fit = None
            if brand_emb is not None:
                w_bc, w_bq, w_qc = P["fit_weights"]
                combined = (w_bc * float(Cr[j] @ brand_emb) + w_bq * float(Cq[i] @ brand_emb)
                            + w_qc * sim)
                fit = round(1 / (1 + np.exp(-P["fit_steepness"] * (combined - P["fit_midpoint"]))), 3)
            pairs.append(dict(
                r_cluster=rid, r_label=rd.loc[rid, "label"],
                r_comments=rd.loc[rid, "n_comments"], r_score=rd.loc[rid, "total_score"],
                q_cluster=qid, q_label=qd.loc[qid, "label"], q_clicks=qd.loc[qid, "clicks"],
                cosine=round(sim, 3), matched=sim >= P["match_threshold"],
                opportunity_score=round(sim * q_demand[qid] * r_signal[rid], 4),
                ranking_queries=qd.loc[qid, "top_queries"],
                community_angle=rd.loc[rid, "representative_comment"],
                brand_fit_probability=fit,
            ))
    pairs = pd.DataFrame(pairs)

    out_cols = ["r_label", "q_label", "q_clicks", "cosine", "opportunity_score",
                "ranking_queries", "community_angle"]
    if brand_emb is not None:
        out_cols.append("brand_fit_probability")
    opps = (pairs[pairs.matched].sort_values("opportunity_score", ascending=False)
            .reset_index(drop=True)[out_cols])

    # gaps
    r_gaps = r_sum[r_sum.r_cluster.map(lambda c: S[:, r_ids.index(c)].max() < P["match_threshold"])].copy()
    r_gaps["best_query_cluster"] = r_gaps.r_cluster.map(
        lambda c: qd.loc[q_ids[S[:, r_ids.index(c)].argmax()], "label"])
    r_gaps["best_cosine"] = r_gaps.r_cluster.map(lambda c: round(float(S[:, r_ids.index(c)].max()), 3))
    r_gaps = r_gaps.drop(columns="signal")

    q_gaps = q_sum[q_sum.q_cluster.map(lambda c: S[q_ids.index(c)].max() < P["match_threshold"])].copy()
    q_gaps["best_reddit_cluster"] = q_gaps.q_cluster.map(
        lambda c: rd.loc[r_ids[S[q_ids.index(c)].argmax()], "label"])
    q_gaps["best_cosine"] = q_gaps.q_cluster.map(lambda c: round(float(S[q_ids.index(c)].max()), 3))

    # query-level matches
    Sqc = Eq @ Er.T
    top = np.argsort(-Sqc, axis=1)[:, :P["top_comments"]]
    q_lvl = []
    for i, row in q.iterrows():
        idx = top[i]
        best = float(Sqc[i, idx[0]])
        q_lvl.append(dict(
            query=row["text"], clicks=int(row["weight"]),
            q_cluster=row["cluster"], best_r_cluster=int(r.iloc[idx[0]]["cluster"]),
            top_cosine=round(best, 3),
            community_match=best >= P["query_comment_threshold"],
            opportunity_score=round(np.log1p(row["weight"]) * best, 3),
            matching_comments=" || ".join(trunc(r.iloc[k]["text"], 180) for k in idx),
        ))
    q_lvl = pd.DataFrame(q_lvl).sort_values("opportunity_score", ascending=False).reset_index(drop=True)

    return dict(stats=stats, sim_stats=sim_stats, q_sum=q_sum, r_sum=r_sum.drop(columns="signal"),
                opps=opps, r_gaps=r_gaps, q_gaps=q_gaps, q_lvl=q_lvl,
                brand=P["brand_name"] if brand_emb is not None else None)


def to_excel(res):
    buf = io.BytesIO()
    sheets = {"opportunities": res["opps"], "reddit_clusters": res["r_sum"],
              "query_clusters": res["q_sum"], "reddit_gaps": res["r_gaps"],
              "query_gaps": res["q_gaps"], "query_level": res["q_lvl"]}
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
st.title("Reddit thread vs GSC queries: content opportunities")

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

    with st.expander("Embedding"):
        MODEL_NAME = st.selectbox("Model", ["all-mpnet-base-v2", "all-MiniLM-L6-v2"])
        MAX_SEQ_LEN = st.number_input("Max sequence length", 32, 512, 256, step=32)

    with st.expander("Query pre-processing"):
        TYPO_MODE = st.selectbox("Typo handling", ["drop", "correct", "off"])
        TYPO_MAX_FREQ = st.number_input("Typo candidate max token frequency", 1, 20, 1)
        ALLOW = st.text_input("Allowlist (comma-separated)", "marella, douro, tui")

    with st.expander("Reddit pre-processing"):
        BRANDS = st.text_input("Brands to remove (comma-separated)", "")
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

    with st.expander("Brand fit (optional)"):
        BRAND_NAME = st.text_input("Brand to score (blank = skip)", "")
        BRAND_SPECS = st.text_area("Brand specs: products, destinations, audience, USPs", "")
        w1 = st.slider("Weight: brand ↔ community topic", 0.0, 1.0, 0.4, 0.05)
        w2 = st.slider("Weight: brand ↔ ranking query cluster", 0.0, 1.0, 0.3, 0.05)
        w3 = st.slider("Weight: query ↔ community topic", 0.0, 1.0, 0.3, 0.05)
        FIT_MID = st.slider("Fit midpoint", 0.0, 1.0, 0.35, 0.01)
        FIT_STEEP = st.slider("Fit steepness", 1, 30, 10)

    with st.expander("Comparison"):
        MATCH_T = st.slider("Match threshold (centroid cosine)", 0.0, 1.0, 0.35, 0.01)
        QC_T = st.slider("Query ↔ comment threshold", 0.0, 1.0, 0.30, 0.01)
        TOP_Q = st.number_input("Query clusters per Reddit cluster", 1, 10, 2)
        TOP_C = st.number_input("Comments per query", 1, 10, 3)

    run = st.button("Run analysis", type="primary", disabled=not (q_file and r_file))

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
            fit_weights=(w1 / wsum, w2 / wsum, w3 / wsum), fit_midpoint=FIT_MID,
            fit_steepness=FIT_STEEP, match_threshold=MATCH_T, query_comment_threshold=QC_T,
            top_q_per_r=int(TOP_Q), top_comments=int(TOP_C), brand_name=BRAND_NAME.strip(),
        )
        with st.status("Running analysis…", expanded=True) as status:
            st.write("Pre-processing…")
            q = q_df[[QUERY_COL, CLICKS_COL]].dropna(subset=[QUERY_COL]).copy()
            q.columns = ["text", "weight"]
            allow = {w.strip().lower() for w in ALLOW.split(",") if w.strip()}
            q, q_log = prep_queries(q, TYPO_MODE, allow, int(TYPO_MAX_FREQ))

            r = r_df[[REDDIT_COL, SCORE_COL]].dropna(subset=[REDDIT_COL]).copy()
            r.columns = ["text", "weight"]
            brands = [b.strip() for b in BRANDS.split(",") if b.strip()]
            r, r_log = prep_reddit(r, brands, BRAND_MODE, int(MIN_WORDS), int(MIN_CHARS),
                                   MIN_ALPHA, DROP_REGEX.strip() or None)
            if len(q) < 3 or len(r) < 3:
                raise ValueError(f"Too few rows after cleaning (queries={len(q)}, comments={len(r)}).")

            st.write(f"Embedding {len(q)} queries and {len(r)} comments (first run downloads the model)…")
            Eq = embed(MODEL_NAME, int(MAX_SEQ_LEN), tuple(q["text"]))
            Er = embed(MODEL_NAME, int(MAX_SEQ_LEN), tuple(r["text"]))
            brand_emb = None
            if P["brand_name"]:
                brand_emb = embed(MODEL_NAME, int(MAX_SEQ_LEN),
                                  (f"{P['brand_name']}. {BRAND_SPECS.strip()}",))[0]

            st.write("Clustering and comparing…")
            res = analyse(q, r, Eq, Er, brand_emb, P)
            res.update(q_log=q_log, r_log=r_log, n_q=len(q), n_r=len(r))
            st.session_state["res"] = res
            status.update(label="Done", state="complete", expanded=False)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------ results
res = st.session_state.get("res")
if not res:
    st.info("Upload both files, set columns and options in the sidebar, then click **Run analysis**.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Queries (clean)", res["n_q"])
c2.metric("Reddit comments (clean)", res["n_r"])
c3.metric("Matched opportunities", len(res["opps"]))

st.download_button(
    "Download results (xlsx)", to_excel(res), "content_opportunities.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

tabs = st.tabs(["Opportunities", "Reddit gaps", "Query gaps", "Query-level",
                "Query clusters", "Reddit clusters", "Diagnostics"])

with tabs[0]:
    if res["opps"].empty:
        st.warning("No matches above the threshold. Lower the match threshold in the sidebar.")
    else:
        if res["brand"]:
            st.caption(f"brand_fit_probability is scored for {res['brand']}. "
                       "Tune the fit midpoint if values are all high or all low.")
        st.dataframe(res["opps"], width="stretch", hide_index=True)

with tabs[1]:
    st.caption("Community topics with no ranking-query match: new content / keyword ideas.")
    st.dataframe(res["r_gaps"][["r_cluster", "label", "n_comments", "total_score", "best_query_cluster",
                                "best_cosine", "representative_comment"]],
                 width="stretch", hide_index=True)

with tabs[2]:
    st.caption("Ranking queries with no community match.")
    st.dataframe(res["q_gaps"].drop(columns="top_queries"), width="stretch", hide_index=True)

with tabs[3]:
    only_match = st.checkbox("Only community matches", False)
    d = res["q_lvl"]
    d = d[d.community_match] if only_match else d
    st.dataframe(d, width="stretch", hide_index=True)

with tabs[4]:
    st.dataframe(res["q_sum"], width="stretch", hide_index=True)

with tabs[5]:
    st.dataframe(res["r_sum"], width="stretch", hide_index=True)

with tabs[6]:
    st.subheader("Cluster stats")
    st.dataframe(res["stats"], width="stretch", hide_index=True)
    st.subheader("Centroid cosine distribution (use to tune the match threshold)")
    st.dataframe(res["sim_stats"], width="stretch", hide_index=True)
    st.subheader("Query pre-processing")
    st.write("\n".join(f"- {l}" for l in res["q_log"]))
    st.subheader("Reddit pre-processing")
    st.dataframe(res["r_log"], width="stretch", hide_index=True)
