**Link to full AI Chat Log:** [https://chat.deepseek.com/share/58lfgwb1ynbnpnkplx]

# AI Assistant Transcript - Phase 1

Below is a summary of the prompts and assistance provided by the AI assistant during the development of Phase 1.

**User:** Explain the requirements for building a corpus and a hybrid retrieval system.
**AI:** Provided a breakdown of the corpus requirements (400+ papers, 4 sub-topics, short/medium/long variants) and retrieval engine requirements (BM25, dense, fusion, spellcheck, GUI).

**User:** Give me an arXiv category that covers at least 4 distinct sub-topics.
**AI:** Suggested cs.CL (Computation and Language) and proposed Machine Translation, Question Answering, Summarization, and Dialogue Systems.

**User:** Write the code for the full system.
**AI:** Generated `collect_corpus.py`, `build_index.py`, `retriever.py`, `app.py`, `report_stats.py`, and `requirements.txt`. It also provided instructions for setting up the virtual environment and Git.

**User:** Help debug the corpus collection script (returning 0 papers for some sub-topics).
**AI:** Identified that the arXiv API was strict about exact phrase matching and date ranges. Suggested broadening the queries and removing the date filter to pull the latest 480 papers.

**User:** Help with Streamlit and GitHub.
**AI:** Explained how to fix the Conda/venv conflict (`python -m streamlit run app.py`) and provided the exact Git commands to push the project to GitHub.

*All code provided by the AI was executed, tested, and verified by the user.*