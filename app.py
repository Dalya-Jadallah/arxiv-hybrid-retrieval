import streamlit as st
from retriever import Retriever

st.set_page_config(page_title="arXiv cs.CL Hybrid Search", layout="wide")

@st.cache_resource
def load_retriever():
    return Retriever.load("data")

ret = load_retriever()

st.title("🔎 arXiv cs.CL — Hybrid Retrieval")
st.caption("BM25 + Dense fusion · query preprocessing: spell-check, abbreviation, case, phrase boost")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Search query",
                          placeholder='e.g. "neural machine translation" QA')
with col2:
    fusion = st.selectbox(
        "Fusion / model",
        ["rrf", "weighted", "bm25", "dense"],
        format_func=lambda x: {
            "rrf": "RRF (BM25 + Dense)",
            "weighted": "Weighted (0.5/0.5)",
            "bm25": "BM25 only",
            "dense": "Dense only",
        }[x],
    )

top_k = st.slider("Number of results", 5, 20, 10)

if query:
    # Fetch 50 to ensure we have enough unique papers after deduplication
    results, pre = ret.search(query, fusion=fusion, top_k=50)
    
    # Deduplicate by arxiv_id, keeping the first (highest-ranked) variant
    seen_papers = set()
    unique_results = []
    for r in results:
        if r['arxiv_id'] not in seen_papers:
            seen_papers.add(r['arxiv_id'])
            unique_results.append(r)
    
    # Now trim to the user's requested top_k
    results = unique_results[:top_k]

    st.markdown("**Query preprocessing**")
    st.code(f"clean: {pre['clean']}\nphrases: {pre['phrases']}\nsearch: {pre['search_text']}")

    st.markdown(f"### Top {len(results)} results")
    if not results:
        st.warning("No results.")
    for i, r in enumerate(results, 1):
        st.markdown(f"#### {i}. [{r['title']}]({r['link']})")
        st.caption(
            f"Score: `{r['score']:.4f}` · Sub-topic: **{r['subtopic']}** · "
            f"Variant: **{r['variant']}** · arXiv: `{r['arxiv_id']}`"
        )
        st.write(r["snippet"])
        st.divider()
else:
    st.info("Enter a query above to search.")