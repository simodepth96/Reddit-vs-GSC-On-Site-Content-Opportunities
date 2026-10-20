# Reddit vs GSC: Content Opportunities

A framework that compares Reddit threads with your Google Search Console queries and scores each match as a content opportunity for a brand. Clusters are named by Gemini, their entities are confirmed in the Google Knowledge Graph, and TypeSafe's Jev is used in combination with cosine similarity to score the probability that your brand should target Reddit-based topics you are not yet ranking for.

## Pipeline

1. **Cleans queries:** lower-cases, drops non-ASCII queries, merges duplicates (clicks summed), and drops or corrects typos.
2. **Cleans Reddit comments:** removes deleted/removed comments, optionally removes brand mentions, drops pleasantries, and filters by word count, character count and alpha ratio.
3. **Embeds** both sets with a sentence-transformers model.
4. **Clusters** each set with HDBSCAN (optional UMAP and noise absorption).
5. **Matches** Reddit clusters to query clusters based on cosine similarity.
6. **Names clusters (Gemini API):** renames every Reddit and GSC cluster in natural language and proposes the Google entity each one is about.
7. **Confirms entities (Knowledge Graph Search API):** looks up each proposed entity. A result is accepted only if its name matches, so each cluster ends up with a confirmed entity or none.
8. **Scores (TypeSafe Jev API):** for each matched pair, Jev receives the brand profile, both cluster names, their confirmed entities, clicks and example text, and answers three yes/no probabilities:
   - `brand_fit`: is the brand a credible publisher on this topic, and is it not out of scope?
   - `entity_match`: do both topics concern the same confirmed entity (or closely related ones)? Topics with no confirmed entity are treated as unvalidated.
   - `content_value`: would new content answer unmet community questions and capture winnable search demand?
   `content_opportunity_score` is the weighted average of the three (weights set in the sidebar, normalised to sum to 1).
9. **Finds gaps in both directions:** community topics with no ranking-query match, and ranking queries with no community match.

Cosine similarity is used internally for matching and gap detection only. It is not shown in the output.

### API keys

| Service | Key | Notes |
|---|---|---|
| Gemini API | `GEMINI_API_KEY` | Google AI Studio key. Default model alias is `gemini-3.8-flash` (editable) |
| Knowledge Graph Search API | `GOOGLE_KG_API_KEY` | Enable "Knowledge Graph Search API" in your Google Cloud project and create an API key |
| TypeSafe (Jev) | `TYPESAFE_API_KEY` | Create the key in the TypeSafe console. Default model is `jev-latest`; pin a version (e.g. `jev-1.13.0`) for reproducible scores |

Keys are read from environment variables or `.streamlit/secrets.toml`, or typed into the sidebar:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "..."
GOOGLE_KG_API_KEY = "..."
TYPESAFE_API_KEY = "..."
```

Responses from all three APIs are cached for the life of the Streamlit process. Re-running with different clustering thresholds only calls the APIs for new inputs.

## Input files

Both files can be `.xlsx`, `.xls` or `.csv`. Column names are chosen from dropdowns after upload.

| File | Needs | Default columns |
|---|---|---|
| GSC queries | query text, clicks | `Query`, `Clicks` |
| Reddit comments | comment text, score | `keyword`, `score` |

## Usage

1. Upload both files and confirm the columns.
2. Enter the API keys and the brand (name, specs, optional out-of-scope topics).
3. Adjust other settings if needed.
4. Click **Run analysis**. The button stays disabled until files, brand and keys are all set.

## Sidebar settings

| Group | Setting | Notes |
|---|---|---|
| API keys | Keys, Gemini model, Jev model | See above |
| | Parallel API calls | Threads for Knowledge Graph and Jev calls |
| Brand | Name, specs | Specs: products, destinations, audience, USPs. The more specific, the better the brand-fit probability |
| | Out of scope topics | Topics the brand must not publish on; Jev uses them in `brand_fit` |
| | Score weights | Brand fit / Entity match / Content value (default 0.4 / 0.3 / 0.3) |
| Embedding | Model, max sequence length | `all-mpnet-base-v2` is more accurate; `all-MiniLM-L6-v2` is faster |
| Query pre-processing | Typo handling, max token frequency, allowlist | Add place and brand names to the allowlist |
| Reddit pre-processing | Brands to remove, brand mode | `drop_rows` removes the comment; `strip_term` deletes only the brand word |
| | Min words / chars / alpha ratio, drop regex | Run after cleaning and brand removal |
| HDBSCAN (queries, Reddit) | min_cluster_size, min_samples, method, UMAP, noise absorb | Tune these first |
| Matching | Match threshold | Centroid-to-centroid cosine for a topical match |
| | Query clusters per Reddit cluster | Candidate matches per Reddit cluster |
| | Max pairs to score | Caps Jev calls; the closest pairs are scored first |

## Output

Tabs and sheets in `content_opportunities.xlsx`:

| Tab / sheet | Columns |
|---|---|
| Opportunities (`opportunities`) | `Reddit Cluster`, `GSC Cluster`, `Confirmed Entity`, `q_clicks`, `ranking_queries`, `community_angle`, `content_opportunity_score` |
| Reddit gaps (`reddit_gaps`) | `Reddit Cluster`, `Confirmed Entity`, `n_comments`, `representative_comment`, `Closest GSC Cluster` |
| Query gaps (`query_gaps`) | `GSC Cluster`, `Confirmed Entity`, `n_queries`, `clicks`, `top_queries`, `Closest Reddit Cluster` |
| Query clusters (`query_clusters`) | `GSC Cluster`, `Confirmed Entity`, `n_queries`, `clicks`, `top_queries` |
| Reddit clusters (`reddit_clusters`) | `Reddit Cluster`, `Confirmed Entity`, `n_comments`, `representative_comment` |

`Confirmed Entity` is the Knowledge Graph name and type. It is blank when no match was confirmed.

## Tuning tips

- **Few or no matches:** lower the match threshold or raise "Query clusters per Reddit cluster".
- **Too much noise:** lower `min_cluster_size` / `min_samples`, enable noise absorb, or switch the method to `eom` for fewer, larger clusters.
- **Scores all high or all low:** make the brand specs and out-of-scope list more specific, or shift the weights. Weights are applied after the Jev calls, so changing them is free.
- **Low "clusters with confirmed entity" count:** many clusters are generic topics with no Knowledge Graph entity. Those pairs score low on `entity_match` by design.

## Troubleshooting

| Message | Fix |
|---|---|
| `Gemini naming failed … keyword labels used instead` | Check the key and model name. The run continues with keyword labels and no entities |
| `Knowledge Graph lookups failed for N entities` | Check that the Knowledge Graph Search API is enabled for the key's project, and its quota |
| `TypeSafe Jev failed: …` | Check the key and model. 429/5xx responses are retried automatically; if every pair fails, the run stops |
| `Jev failed for N of M pairs` | Affected rows have an empty score; re-run, since successful pairs are cached |
| `HDBSCAN found no clusters` | Lower `min_cluster_size` / `min_samples`; everything is put in one cluster so the run continues |
| `Too few rows after cleaning` | Relax the Reddit filters or the typo handling |
