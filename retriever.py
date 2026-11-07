"""
Hybrid retriever: BM25 + FAISS dense + RRF / weighted fusion,
with query preprocessing (spell, abbreviation, case, phrase boost).
"""
import json
import pickle
import re
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker

ABBREV = {
    "nlp": "natural language processing",
    "mt":  "machine translation",
    "qa":  "question answering",
    "ir":  "information retrieval",
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "llms":"large language models",
    "sota":"state of the art",
    "pos": "part of speech",
    "ner": "named entity recognition",
}

class Retriever:
    def __init__(self, docs, bm25_state, dense_index, dense_meta, embedder, spell):
        self.docs = {d["doc_id"]: d for d in docs}
        self.bm25 = bm25_state["bm25"]
        self.bm25_doc_ids = bm25_state["doc_ids"]
        self.dense_index = dense_index
        self.dense_doc_ids = dense_meta["doc_ids"]
        self.embedder = embedder
        self.spell = spell

    # ---------- loading ----------
    @classmethod
    def load(cls, data_dir="data"):
        with open(f"{data_dir}/documents.jsonl") as f:
            docs = [json.loads(line) for line in f]

        with open(f"{data_dir}/bm25.pkl", "rb") as f:
            bm25_state = pickle.load(f)

        dense_index = faiss.read_index(f"{data_dir}/dense_index.faiss")
        with open(f"{data_dir}/dense_meta.json") as f:
            dense_meta = json.load(f)

        embedder = SentenceTransformer(dense_meta["model"])

        # spell-checker with corpus vocabulary injected
        spell = SpellChecker()
        vocab = set()
        for d in docs:
            for tok in d["text"].lower().split()[:200]:
                if tok.isalpha() and len(tok) > 2:
                    vocab.add(tok)
        spell.word_frequency.load_words(list(vocab)[:50_000])

        return cls(docs, bm25_state, dense_index, dense_meta, embedder, spell)

    # ---------- query preprocessing ----------
    def preprocess(self, q: str):
        q = q.strip().lower()

        # 1) pull out quoted phrases
        phrases = re.findall(r'"([^"]+)"', q)
        q_no_phrase = re.sub(r'"[^"]+"', " ", q)

        # 2) tokenize and expand/correct
        tokens = re.findall(r"[a-zA-Z0-9]+", q_no_phrase)
        fixed = []
        for t in tokens:
            if t in ABBREV:
                fixed.extend(ABBREV[t].split())
            else:
                c = self.spell.correction(t)
                fixed.append(c if c else t)

        clean = " ".join(fixed)
        # append phrase tokens so BM25/dense see them too
        phrase_tokens = " ".join(p for p in phrases)
        search_text = (clean + " " + phrase_tokens).strip()

        return {"clean": clean, "phrases": phrases, "search_text": search_text}

    # ---------- BM25 ----------
    def bm25_search(self, text: str, k=100):
        toks = text.lower().split()
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        top = np.argsort(scores)[::-1][:k]
        return [(self.bm25_doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    # ---------- Dense ----------
    def dense_search(self, text: str, k=100):
        qv = self.embedder.encode([text], normalize_embeddings=True).astype("float32")
        scores, idxs = self.dense_index.search(qv, k)
        return [(self.dense_doc_ids[i], float(s))
                for s, i in zip(scores[0], idxs[0]) if i >= 0]

    # ---------- Fusion ----------
    def fuse_rrf(self, bm25_res, dense_res, k_const=60):
        scores = {}
        for rank, (doc_id, _) in enumerate(bm25_res, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_const + rank)
        for rank, (doc_id, _) in enumerate(dense_res, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_const + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    def fuse_weighted(self, bm25_res, dense_res, alpha=0.5):
        def norm(res):
            if not res:
                return {}
            vals = [s for _, s in res]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return {d: 1.0 for d, _ in res}
            return {d: (s - lo) / (hi - lo) for d, s in res}

        nb, nd = norm(bm25_res), norm(dense_res)
        all_ids = set(nb) | set(nd)
        scores = {d: alpha * nb.get(d, 0.0) + (1 - alpha) * nd.get(d, 0.0)
                  for d in all_ids}
        return sorted(scores.items(), key=lambda x: -x[1])

    # ---------- Phrase boost ----------
    def apply_phrase_boost(self, ranked, phrases, boost=0.20):
        if not phrases:
            return ranked
        out = []
        for doc_id, score in ranked:
            text = self.docs[doc_id]["text"].lower()
            hits = sum(1 for p in phrases if p.lower() in text)
            out.append((doc_id, score * (1 + boost * hits)))
        out.sort(key=lambda x: -x[1])
        return out

    # ---------- Snippet ----------
    def make_snippet(self, doc_id, query_terms, length=200):
        text = self.docs[doc_id]["text"]
        if len(text) <= length:
            return text
        # try to center snippet around first matching query term
        low = text.lower()
        pos = -1
        for t in query_terms:
            p = low.find(t.lower())
            if p >= 0:
                pos = p
                break
        if pos < 0:
            return text[:length].rstrip() + "…"
        start = max(0, pos - length // 3)
        end = min(len(text), start + length)
        snip = text[start:end].strip()
        return ("…" if start > 0 else "") + snip + ("…" if end < len(text) else "")

    # ---------- Main entry ----------
    def search(self, query: str, fusion="rrf", top_k=10):
        pre = self.preprocess(query)
        stext = pre["search_text"] or pre["clean"] or query

        if fusion == "bm25":
            ranked = self.bm25_search(stext, k=top_k * 3)
        elif fusion == "dense":
            ranked = self.dense_search(stext, k=top_k * 3)
        else:
            bm25_res = self.bm25_search(stext, k=100)
            dense_res = self.dense_search(stext, k=100)
            if fusion == "rrf":
                ranked = self.fuse_rrf(bm25_res, dense_res)
            elif fusion == "weighted":
                ranked = self.fuse_weighted(bm25_res, dense_res, alpha=0.5)
            else:
                raise ValueError(f"Unknown fusion: {fusion}")

        ranked = self.apply_phrase_boost(ranked, pre["phrases"])
        ranked = ranked[:top_k]

        terms = (pre["clean"].split() + pre["phrases"])
        results = []
        for doc_id, score in ranked:
            d = self.docs[doc_id]
            results.append({
                "doc_id":   doc_id,
                "arxiv_id": d["arxiv_id"],
                "link":     d["link"],
                "title":    d["title"],
                "subtopic": d["subtopic"],
                "variant":  d["variant"],
                "snippet":  self.make_snippet(doc_id, terms, length=200),
                "score":    float(score),
            })
        return results, pre