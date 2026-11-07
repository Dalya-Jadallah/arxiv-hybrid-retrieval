"""
Generate Phase 1 summary report (PDF).
Reads corpus_raw.json + corpus_manifest.json + documents.jsonl.
Outputs report/phase1_report.pdf.
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib import colors

OUT_DIR = "report"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open("data/corpus_raw.json") as f:
        papers = json.load(f)
    with open("data/corpus_manifest.json") as f:
        manifest = json.load(f)

    docs = []
    try:
        with open("data/documents.jsonl") as f:
            docs = [json.loads(l) for l in f]
    except FileNotFoundError:
        pass

    # ---- stats ----
    subtopic_counts = Counter(p["subtopic"] for p in papers)
    variant_counts = Counter(d["variant"] for d in docs)
    variant_avg_len = {}
    for v in ("short", "medium", "long"):
        lens = [len(d["text"]) for d in docs if d["variant"] == v]
        variant_avg_len[v] = (sum(lens) / len(lens)) if lens else 0
    full_text_papers = sum(1 for p in papers
                           if any(d["arxiv_id"] == p["arxiv_id"] and d["had_full_text"]
                                  for d in docs)) if docs else 0

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(list(subtopic_counts.keys()), list(subtopic_counts.values()),
           color="#4C72B0")
    ax.set_ylabel("# papers")
    ax.set_title("Papers per sub-topic")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p1 = f"{OUT_DIR}/subtopics.png"
    plt.savefig(p1, dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(list(variant_counts.keys()), list(variant_counts.values()),
           color="#55A868")
    ax.set_ylabel("# documents")
    ax.set_title("Documents per length variant")
    plt.tight_layout()
    p2 = f"{OUT_DIR}/variants.png"
    plt.savefig(p2, dpi=150); plt.close()

    # ---- PDF ----
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Phase 1 — Corpus & Hybrid Retrieval Report", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("arXiv category: <b>cs.CL</b>", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("1. Corpus summary", styles["Heading2"]))
    story.append(Paragraph(
        f"Total papers collected: <b>{len(papers)}</b><br/>"
        f"Sub-topics: <b>{len(subtopic_counts)}</b><br/>"
        f"Date range: <b>{manifest['date_range']['from']} "
        f"→ {manifest['date_range']['to']}</b><br/>"
        f"Total indexed documents (3 variants × papers): <b>{len(docs)}</b><br/>"
        f"Papers with full HTML text successfully fetched: "
        f"<b>{full_text_papers}/{len(papers)}</b>",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("2. Sub-topic distribution", styles["Heading2"]))
    data = [["Sub-topic", "Papers", "arXiv query"]]
    for label, info in manifest["per_subtopic"].items():
        data.append([label, str(subtopic_counts.get(label, 0)), Paragraph(info["query"], styles["Normal"])])
    tbl = Table(data, colWidths=[4 * cm, 2 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(p1, width=14 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("3. Length variants", styles["Heading2"]))
    story.append(Paragraph(
        "Each paper is indexed three times: <b>short</b> (title + abstract, "
        "≤1,500 chars), <b>medium</b> (up to 6,000 chars), <b>long</b> "
        "(up to 40,000 chars, full HTML text when available).",
        styles["Normal"],
    ))
    data2 = [["Variant", "Documents", "Avg length (chars)"]]
    for v in ("short", "medium", "long"):
        data2.append([v, str(variant_counts.get(v, 0)),
                      f"{variant_avg_len[v]:.0f}"])
    tbl2 = Table(data2, colWidths=[4 * cm, 4 * cm, 6 * cm])
    tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#55A868")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
    ]))
    story.append(tbl2)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(p2, width=14 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("4. Retrieval design", styles["Heading2"]))
    story.append(Paragraph(
        "<b>Sparse:</b> BM25Okapi (k1=1.5, b=0.75) via rank_bm25.<br/>"
        "<b>Dense:</b> all-MiniLM-L6-v2 embeddings + FAISS IndexFlatIP "
        "(cosine similarity on normalized vectors).<br/>"
        "<b>Fusion:</b> Reciprocal Rank Fusion with k=60, plus an optional "
        "weighted (α=0.5) min-max normalized fusion for comparison.<br/>"
        "<b>Query handling:</b> lowercase normalization; corpus-backed spell "
        "correction (pyspellchecker with domain vocabulary); abbreviation "
        "expansion (NLP→natural language processing, QA→question answering, "
        "MT→machine translation, etc.); phrase extraction for quoted spans, "
        "with a post-fusion phrase-boost multiplier (1 + 0.2·hits).",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("5. Reproducibility", styles["Heading2"]))
    story.append(Paragraph(
        "All data generated with the exact queries in Table 2, "
        "over the stated date range, using arXiv's public API and "
        "HTML endpoints (no authentication). To reproduce:<br/><br/>"
        "<font face='Courier'>pip install -r requirements.txt</font><br/>"
        "<font face='Courier'>python collect_corpus.py</font><br/>"
        "<font face='Courier'>python build_index.py</font><br/>"
        "<font face='Courier'>streamlit run app.py</font>",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("6. AI usage disclosure", styles["Heading2"]))
    story.append(Paragraph(
        "See <font face='Courier'>/ai-logs/phase1/</font>. All code was "
        "verified by running it live prior to submission.",
        styles["Normal"],
    ))

    doc = SimpleDocTemplate(f"{OUT_DIR}/phase1_report.pdf", pagesize=A4,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    doc.build(story)
    print(f"✅ Report → {OUT_DIR}/phase1_report.pdf")


if __name__ == "__main__":
    main()