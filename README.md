# arXiv cs.CL Hybrid Retrieval

Hybrid search (BM25 + dense + fusion) over 480 cs.CL papers across 4 sub-topics:
machine translation, question answering, summarization, dialogue systems.

## Quick start

```bash
pip install -r requirements.txt

# 1. Collect corpus (~3 min, hits arXiv API)
python collect_corpus.py

# 2. Build variants + indexes (~5-10 min)
python build_index.py

# 3. Generate PDF report
python report_stats.py

# 4. Launch GUI
python -m streamlit run app.py