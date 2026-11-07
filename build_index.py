"""
Build three length variants per paper, then BM25 + dense indexes.
- short  : title + abstract
- medium : title + abstract + first ~6,000 chars of full text
- long   : title + full extracted text (or full HTML-stripped text)

Full text is fetched from arXiv HTML (https://arxiv.org/html/{id}).
If HTML isn't available (older papers), falls back to abstract-only,
so long variant == medium for those; that's documented in the report.
"""
import json
import os
import re
import time
import pickle
import numpy as np
import requests
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

SHORT_LEN  = 1_500
MEDIUM_LEN = 6_000
LONG_LEN   = 40_000
HTML_TIMEOUT = 20

# ---------- full-text fetch ----------

def fetch_full_text(arxiv_id: str) -> str:
    """Fetch arXiv HTML view and strip tags. Returns '' if unavailable."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        r = requests.get(url, timeout=HTML_TIMEOUT,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot)"})
        if r.status_code != 200:
            return ""
        html = r.text
        # crude tag strip
        html = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
        html = re.sub(r"<style.*?</style>",  " ", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""

# ---------- variant construction ----------

def build_variants(paper, full_text=""):
    title = paper["title"]
    abstract = paper["abstract"]
    base = f"{title}. {abstract}"
    short = base[:SHORT_LEN]

    if full_text:
        medium = (base + " " + full_text)[:MEDIUM_LEN]
        long   = (base + " " + full_text)[:LONG_LEN]
    else:
        # fall back: medium/long both = short (flag in `variant_note`)
        medium = short
        long   = short

    return {
        "short":  short,
        "medium": medium,
        "long":   long,
        "had_full_text": bool(full_text),
    }

def main():
    with open("data/corpus_raw.json") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")

    documents = []
    fetch_failures = 0
    for p in tqdm(papers, desc="Fetching full text & building variants"):
        full = fetch_full_text(p["arxiv_id"])
        if not full:
            fetch_failures += 1
        variants = build_variants(p, full)
        for variant in ("short", "medium", "long"):
            documents.append({
                "doc_id":     f"{p['arxiv_id']}__{variant}",
                "arxiv_id":   p["arxiv_id"],
                "link":       p["link"],
                "title":      p["title"],
                "text":       variants[variant],
                "subtopic":   p["subtopic"],
                "published":  p["published"],
                "variant":    variant,
                "had_full_text": variants["had_full_text"],
            })
        time.sleep(0.3)  # be polite to arXiv

    print(f"Built {len(documents)} documents "
          f"(fetch failures: {fetch_failures}/{len(papers)})")

    with open("data/documents.jsonl", "w") as f:
        for d in documents:
            f.write(json.dumps(d) + "\n")

    # ---------- BM25 ----------
    print("Tokenizing for BM25 ...")
    tokenized = [d["text"].lower().split() for d in documents]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    with open("data/bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": [d["doc_id"] for d in documents]}, f)
    print("✅ BM25 saved")

    # ---------- Dense ----------
    print("Encoding dense embeddings (this takes a few minutes) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["text"] for d in documents]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")

    np.save("data/dense_embeddings.npy", embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine on normalized vecs
    index.add(embeddings)
    faiss.write_index(index, "data/dense_index.faiss")

    with open("data/dense_meta.json", "w") as f:
        json.dump({"doc_ids": [d["doc_id"] for d in documents],
                   "model": "all-MiniLM-L6-v2"}, f)
    print("✅ Dense index saved")
    print("Done.")

if __name__ == "__main__":
    main()