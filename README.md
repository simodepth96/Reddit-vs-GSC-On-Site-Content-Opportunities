# Reddit vs GSC: Content Opportunities

Streamlit app that compares Reddit comments with your Google Search Console queries to find content opportunities. It embeds both sets of text, clusters them with HDBSCAN, and matches Reddit topics to ranking-query clusters by cosine similarity.

## What it does

1. **Cleans queries:** lower-cases, drops non-ASCII queries, merges duplicates (clicks summed), and drops or corrects typos.
2. **Cleans Reddit comments:** removes deleted/removed comments, optionally removes brand mentions, drops pleasantries, and filters by word count, character count and alpha ratio.
3. **Embeds** both sets with a sentence-transformers model.
4. **Clusters** each set with HDBSCAN (optional UMAP and noise absorption).
5. **Matches** Reddit clusters to query clusters and scores each pair:
   `opportunity_score = cosine × query demand × Reddit signal`
6. **Optionally scores brand fit**, a 0–1 probability for a brand that would publish the content.
7. **Finds gaps** in both directions: community topics with no ranking-query match, and ranking queries with no community match.
8. **Matches at query level:** the closest Reddit comments for each query.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+ is recommended. `scikit-learn>=1.3` is required for `sklearn.cluster.HDBSCAN`.

`sentence-transformers` installs `torch`. On a CPU-only machine, install the CPU build first to keep the install small:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The embedding model downloads on first run, so the first run is slower.

## Input files

Both files can be `.xlsx`, `.xls` or `.csv`. Column names are chosen from dropdowns after upload.

| File | Needs | Default columns |
|---|---|---|
| GSC queries | query text, clicks | `Query`, `Clicks` |
| Reddit comments | comment text, score | `keyword`, `score` |

## Usage

1. Upload both files in the sidebar and confirm the columns.
2. Adjust settings if needed (see below).
3. Click **Run analysis**.
4. Browse the tabs and download the xlsx.

## Sidebar settings

| Group | Setting | Notes |
|---|---|---|
| Embedding | Model, max sequence length | `all-mpnet-base-v2` is more accurate; `all-MiniLM-L6-v2` is faster |
| Query pre-processing | Typo handling | `drop`, `correct` or `off` |
| | Typo max token frequency | A word is a typo candidate only if it appears in at most N queries |
| | Allowlist | Place and brand names never treated as typos |
| Reddit pre-processing | Brands to remove | Comma-separated |
| | Brand mode | `drop_rows` removes the comment; `strip_term` deletes only the brand word |
| | Min words / chars / alpha ratio | Run after cleaning and brand removal |
| | Drop regex | Pleasantries filter; blank disables it |
| HDBSCAN (queries, Reddit) | min_cluster_size, min_samples, method | Tune these first |
| | UMAP, dims, neighbors | Recommended for Reddit once you have more than about 100 comments |
| | Noise absorb | Pulls noise points into the nearest cluster above the cosine you set |
| Brand fit | Brand name, specs | Blank name skips scoring |
| | Three weights | Brand↔community topic, brand↔query cluster, query↔community topic; normalised to sum to 1 |
| | Midpoint, steepness | Midpoint is the combined cosine mapped to 50%; steepness sharpens the split |
| Comparison | Match threshold | Centroid-to-centroid cosine for a topical match |
| | Query↔comment threshold | For query-level matches |
| | Query clusters per Reddit cluster, comments per query | Output size |

## Output

The **Download results** button gives `content_opportunities.xlsx` with these sheets:

| Sheet | Contents |
|---|---|
| `opportunities` | Matched Reddit topic → query cluster pairs, ranked by `opportunity_score` |
| `reddit_clusters` | Label, comment count, total score, representative comment |
| `query_clusters` | Label, query count, clicks, top queries |
| `reddit_gaps` | Reddit topics with no ranking-query match: new content ideas |
| `query_gaps` | Ranking queries with no community match |
| `query_level` | Closest Reddit comments per query |

The **Diagnostics** tab shows cluster and noise counts, the centroid cosine distribution, and the pre-processing logs.

## Tuning tips

- **Too few or no matches:** check the cosine percentiles in Diagnostics, then lower the match threshold.
- **Too much noise:** lower `min_cluster_size` / `min_samples`, enable noise absorb, or switch the method to `eom` for fewer, larger clusters.
- **All brand-fit values high or low:** move the fit midpoint toward the median combined cosine.
- **Changing only clustering settings or thresholds** re-runs quickly, because embeddings are cached. Changing the model, texts or pre-processing settings re-embeds.

## Troubleshooting

| Error | Fix |
|---|---|
| `HDBSCAN found no clusters` warning | Lower `min_cluster_size` / `min_samples`; everything is put in one cluster so the run continues |
| `Too few rows after cleaning` | Relax the Reddit filters or the typo handling |
| `ImportError: HDBSCAN` | Upgrade `scikit-learn` to 1.3 or later |
