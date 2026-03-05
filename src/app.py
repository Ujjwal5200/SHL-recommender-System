"""
SHL Assessment Recommender - Professional Streamlit App
A clean, intuitive interface for finding SHL assessments.
"""
import streamlit as st
import requests
import pandas as pd
from typing import List, Dict, Optional
import os

# ==================== CONFIG ====================
USE_LOCAL_API = os.environ.get("USE_LOCAL_API", "false").lower() == "true"
LOCAL_API_URL = os.environ.get("LOCAL_API_URL", "http://localhost:8000/recommend")
REMOTE_API_URL = os.environ.get("API_URL", "https://shl-recommender-api-dxe7.onrender.com/recommend")

API_URL = LOCAL_API_URL if USE_LOCAL_API else REMOTE_API_URL
REQUEST_TIMEOUT = 60

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: #a8c5e2;
    }
    
    /* Card styling for results */
    .result-card {
        padding: 1.25rem;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #2d5a87;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    
    .result-url a {
        color: #2d5a87;
        text-decoration: none;
        font-weight: 500;
    }
    
    .result-url a:hover {
        text-decoration: underline;
    }
    
    .result-score {
        display: inline-block;
        background: #2d5a87;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .result-reason {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d5a87;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Example query chips */
    .example-chip {
        display: inline-block;
        padding: 6px 14px;
        background: #e8f0f8;
        color: #2d5a87;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
        cursor: pointer;
        border: 1px solid #d0e0f0;
        transition: all 0.2s;
    }
    
    .example-chip:hover {
        background: #2d5a87;
        color: white;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-local {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-remote {
        background: #d4edda;
        color: #155724;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .status-local .status-dot {
        background: #ffc107;
    }
    
    .status-remote .status-dot {
        background: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_recommendations(query: str, top_k: int = 10, use_rerank: bool = True) -> List[Dict]:
    """Fetch recommendations from the API."""
    payload = {"query": query}
    
    if USE_LOCAL_API:
        payload["top_k"] = top_k
        payload["use_rerank"] = use_rerank
    
    response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    
    if "recommended_assessments" in result:
        return result["recommended_assessments"]
    elif "results" in result:
        return result["results"]
    return result


def display_result_card(item: Dict, index: int):
    """Display a single result as a styled card."""
    name = item.get("name", "Unknown Assessment")
    url = item.get("url", "#")
    score = item.get("score", 0)
    reason = item.get("reason", "")
    
    # Score percentage
    score_pct = int(score * 100) if score <= 1 else int(score)
    
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">{index + 1}. {name}</div>
        <div class="result-url">
            <a href="{url}" target="_blank">🔗 View Assessment →</a>
        </div>
        <span class="result-score">Match: {score_pct}%</span>
        <div class="result-reason">{reason}</div>
    </div>
    """, unsafe_allow_html=True)


def display_example_queries():
    """Display clickable example queries."""
    examples = [
        "Java developer assessment",
        "Python coding test for data scientists",
        "Sales manager personality test",
        "Excel skills assessment for analysts",
        "Technical competency test for engineers",
        "Customer service representative screening",
        "Leadership and management evaluation",
        "Financial analyst quantitative test"
    ]
    
    cols = st.columns(4)
    for i, example in enumerate(examples):
        with cols[i % 4]:
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.query_text = example
                st.rerun()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://www.shl.com/shl1315/wp-content/uploads/2021/12/SHL-logo.png", width=150)
    
    st.markdown("### ⚙️ Settings")
    
    st.markdown("**API Endpoint:**")
    status_class = "status-local" if USE_LOCAL_API else "status-remote"
    status_text = "🔴 Local" if USE_LOCAL_API else "🟢 Cloud"
    st.markdown(f'<div class="status-indicator {status_class}"><span class="status-dot"></span>{status_text}</div>', 
                unsafe_allow_html=True)
    
    with st.expander("Advanced Options", expanded=False):
        top_k = st.slider("Results to show", 5, 10, 3)
        use_rerank = st.checkbox("Use AI Reranking", value=True, help="Uses Gemini to improve relevance")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info("""
    **SHL Assessment Recommender** uses advanced AI to match job requirements with SHL's assessment library.
    
    **Features:**
    • Natural language matching
    • Semantic search
    • AI-powered reranking
    • 500+ assessments
    """)
    
    st.markdown("---")
    st.markdown("**Need Help?** Contact your HR tech team.")

# ==================== MAIN CONTENT ====================
# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">📋 SHL Assessment Recommender</div>
    <div class="header-subtitle">Find the perfect assessments for your hiring needs</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state for selected example
if "selected_example" not in st.session_state:
    st.session_state.selected_example = ""

# Example queries - clickable links
examples = [
    "Java developer assessment",
    "Python coding test for data scientists",
    "Sales manager personality test",
    "Excel skills assessment for analysts",
    "Technical competency test for engineers",
    "Customer service representative screening",
    "Leadership and management evaluation",
    "Financial analyst quantitative test"
]

st.markdown("**💡 Try these examples:**")
example_cols = st.columns(4)
for i, example in enumerate(examples):
    with example_cols[i % 4]:
        if st.button(f"📝 {example[:25]}...", key=f"example_{i}", use_container_width=True):
            st.session_state.selected_example = example

# Query input - use session state for pre-filled value
query = st.text_area(
    "Describe your hiring needs:",
    height=120,
    placeholder="e.g., We need to assess Python developers with strong problem-solving skills...",
    value=st.session_state.selected_example,
    key="query_input"
)

# Search button
col1, col2 = st.columns([1, 4])
with col1:
    search_btn = st.button("🔍 Find Assessments", type="primary", use_container_width=True)
with col2:
    st.markdown(f"*Powered by{' Ollama + Gemini' if use_rerank else ' advanced search'}*")

# ==================== RESULTS ====================
if search_btn and query.strip():
    with st.spinner("🔎 Analyzing requirements and finding best matches..."):
        try:
            results = fetch_recommendations(query, top_k=top_k, use_rerank=use_rerank)
            
            if not results:
                st.warning("""
                ⚠️ No assessments found matching your criteria.
                
                **Suggestions:**
                • Try different keywords
                • Use more general terms
                • Check your spelling
                """)
            else:
                # Success header
                st.success(f"✅ Found {len(results)} relevant assessment(s)")
                
                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(results)}</div>
                        <div class="metric-label">Results Found</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    # Count unique test types
                    test_types = set()
                    for r in results:
                        test_types.update(r.get("test_types", []))
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(test_types)}</div>
                        <div class="metric-label">Test Categories</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    avg_score = sum(r.get("score", 0) for r in results) / len(results)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{int(avg_score * 100)}%</div>
                        <div class="metric-label">Avg. Match Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Results
                st.markdown("### 📊 Recommended Assessments")
                
                for i, item in enumerate(results):
                    display_result_card(item, i)
                    
                # Export option
                if results:
                    df = pd.DataFrame([{
                        "Rank": i + 1,
                        "Assessment": r.get("name", ""),
                        "URL": r.get("url", ""),
                        "Match Score": f"{r.get('score', 0)*100:.0f}%",
                        "Reason": r.get("reason", "")
                    } for i, r in enumerate(results)])
                    
                    st.markdown("---")
                    with st.expander("📥 Download Results"):
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="shl_recommendations.csv",
                            mime="text/csv"
                        )
                        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The server is taking too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Could not connect to the API. Please check your connection.")
        except requests.exceptions.HTTPError as e:
            st.error(f"⚠️ API error: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

elif search_btn and not query.strip():
    st.warning("⚠️ Please enter a query to find assessments.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <small>SHL Assessment Recommender v9.0 • Built with ❤️ using FastAPI, Streamlit & Gemini</small>
</div>
""", unsafe_allow_html=True)

