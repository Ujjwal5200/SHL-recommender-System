"""
Standalone Streamlit App for SHL Assessment Recommender
This version combines the API and UI for easy deployment to Hugging Face Spaces
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommender_optimized import get_recommendations

st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="wide",
    page_icon="📊"
)

st.title("SHL Assessment Recommendation System")

st.markdown(
    """
Enter **natural language**, **job description**, or a **JD URL**  
to receive **5–10 relevant SHL Individual Test Solutions**.
"""
)

query = st.text_area(
    "Query / JD text / URL",
    height=150,
    placeholder="Example: Hiring Java developers with strong collaboration and problem-solving skills..."
)

if st.button("Get Recommendations", type="primary"):
    if not query.strip():
        st.warning("Query cannot be empty")
    else:
        with st.spinner("Finding best assessments..."):
            try:
                results = get_recommendations(query, top_n=10)
                
                if not results:
                    st.info("No recommendations found. Try a more specific query.")
                else:
                    st.success(f"Found {len(results)} relevant assessments")
                    
                    for item in results:
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"### {item.get('name', 'Unknown')}")
                            with col2:
                                st.markdown(f"[View Assessment]({item.get('url', '#')})")
                            
                            cols = st.columns(4)
                            with cols[0]:
                                st.metric("Duration", f"{item.get('duration_minutes', 'N/A')} min")
                            with cols[1]:
                                st.metric("Adaptive", item.get('adaptive_support', 'No'))
                            with cols[2]:
                                st.metric("Remote", item.get('remote_support', 'No'))
                            with cols[3]:
                                st.metric("Score", f"{item.get('score', 0):.1f}")
                            
                            if item.get('description'):
                                st.write(item['description'][:500])
                            st.divider()
                            
            except Exception as e:
                st.error(f"Error: {e}")

# Sidebar with info
st.sidebar.title("About")
st.sidebar.info(
    """
    **SHL Assessment Recommender**
    
    This system recommends relevant SHL assessments based on natural language queries or job descriptions.
    
    **Performance:**
    - Mean Recall@10: 51.00%
    
    **Tech Stack:**
    - FastAPI
    - Streamlit
    - Keyword-based matching
    """
)
