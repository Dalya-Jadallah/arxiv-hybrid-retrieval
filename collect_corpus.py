"""
Collect 480 cs.CL papers across 4 sub-topics from arXiv.
Saves corpus_raw.json and corpus_manifest.json for reproducibility.
"""
import arxiv
import json
import time
import datetime

SUBDOMAIN_QUERIES = {
    "machine_translation": 'cat:cs.CL AND abs:machine AND abs:translation',
    "question_answering":  'cat:cs.CL AND abs:question AND abs:answering',
    "summarization":       'cat:cs.CL AND abs:summarization',
    "dialogue_systems":    'cat:cs.CL AND abs:dialogue',
}
TARGET_PER_SUBTOPIC = 120
DATE_FROM = datetime.date(2023, 1, 1)
DATE_TO   = datetime.date(2026, 6, 30)

def main():
    client = arxiv.Client(page_size=100, delay_seconds=3.0)
    manifest = {
        "category": "cs.CL",
        "date_range": {"from": "latest available", "to": "latest available"},
        "target_per_subtopic": TARGET_PER_SUBTOPIC,
        "per_subtopic": {},
    }
    all_docs = []

    for label, query in SUBDOMAIN_QUERIES.items():
        print(f"\n=== Collecting: {label} ===")
        print(f"Query: {query}")
        search = arxiv.Search(
            query=query,
            max_results=TARGET_PER_SUBTOPIC * 3,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        kept = 0
        try:
            for r in client.results(search):
                all_docs.append({
                    "arxiv_id": r.get_short_id(),
                    "link": r.entry_id,
                    "title": r.title.strip().replace("\n", " "),
                    "abstract": r.summary.strip().replace("\n", " "),
                    "authors": [a.name for a in r.authors],
                    "published": str(r.published.date()),
                    "subtopic": label,
                })
                kept += 1
                if kept >= TARGET_PER_SUBTOPIC:
                    break
        except Exception as e:
            print(f"ERROR in {label}: {e}")
        manifest["per_subtopic"][label] = {
            "query": query,
            "collected": kept,
            "retrieved_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        print(f"Collected {kept} papers for {label}")

    manifest["total_papers"] = len(all_docs)

    with open("data/corpus_raw.json", "w") as f:
        json.dump(all_docs, f, indent=2)
    with open("data/corpus_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Total: {len(all_docs)} papers → data/corpus_raw.json")
    print(f"✅ Manifest → data/corpus_manifest.json")

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()