# SHL Recommender - Unified Streamlit App
# Combines features from both app.py and streamlit_app.py
import streamlit as st
import requests
import pandas as pd
from typing import List, Dict
import os

# ---------------- CONFIG ----------------
# Toggle between local and remote API
USE_LOCAL_API = os.environ.get("USE_LOCAL_API", "false").lower() == "true"
LOCAL_API_URL = os.environ.get("LOCAL_API_URL", "http://localhost:8000/recommend")
REMOTE_API_URL = os.environ.get("REMOTE_API_URL", "https://recommender-system-api-ikgr.onrender.com/recommend")

# Select API URL based on config
if USE_LOCAL_API:
    API_URL = LOCAL_API_URL
else:
    API_URL = REMOTE_API_URL

REQUEST_TIMEOUT = 60

st.set_page_config(
    page_title="SHL Recommender",
    layout="wide",
    page_icon="📊"
)

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def fetch_recommendations(query: str, top_k: int = 10, use_rerank: bool = True) -> List[Dict]:
    payload = {"query": query}
    
    # Add optional parameters for local API
    if USE_LOCAL_API:
        payload["top_k"] = top_k
        payload["use_rerank"] = use_rerank
    
    response = requests.post(
        API_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    result = response.json()
    
    # Handle both response formats
    if "recommended_assessments" in result:
        return result["recommended_assessments"]
    elif "results" in result:
        return result["results"]
    else:
        return result

def validate_data(data: List[Dict]) -> pd.DataFrame:
    required_cols = {
        "name", "url", "test_types",
        "adaptive_support", "remote_support"
    }

    df = pd.DataFrame(data)

    missing = required_cols - set(df.columns)
    if missing:
        # Return what we have if not all columns present
        return df

    return df[list(required_cols)]

# ---------------- UI ----------------
st.title("SHL Assessment Recommender")

# API Status Indicator
api_status = "🔴 Local" if USE_LOCAL_API else "🟢 Remote"
st.markdown(f"**API Status:** {api_status} ({API_URL})")

st.markdown(
    """
Enter **natural language**, **job description**, or a **JD URL**  
to receive **relevant SHL Individual Test Solutions**.
"""
)

query = st.text_area(
    "Query / JD text / URL",
    height=150,
    placeholder="Example: Hiring Java developers with strong collaboration and problem-solving skills..."
)

# Options row
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    top_k = st.slider("Number of recommendations", 5, 20, 10)
with col2:
    use_rerank = st.checkbox("Use Gemini Reranking", value=True)
with col3:
    show_details = st.checkbox("Show detailed descriptions", value=True)

if st.button("Get Recommendations", type="primary"):
    if not query.strip():
        st.warning("Query cannot be empty")
        st.stop()

    with st.spinner("Finding best assessments..."):
        try:
            results = fetch_recommendations(query, top_k=top_k, use_rerank=use_rerank)

            if not results:
                st.info("No recommendations found. Try a more specific query.")
                st.stop()

            # Try to validate and display as dataframe
            try:
                df = validate_data(results)
                
                st.success(f"Found {len(df)} relevant assessments")
                
                # Display dataframe if we have the expected columns
                if all(col in df.columns for col in ["name", "url", "test_types"]):
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "url": st.column_config.LinkColumn(
                                "Assessment URL",
                                display_text="View"
                            ),
                            "test_types": st.column_config.ListColumn("Test Types"),
                            "adaptive_support": st.column_config.CheckboxColumn("Adaptive"),
                            "remote_support": st.column_config.CheckboxColumn("Remote")
                        }
                    )
                else:
                    # Fallback: display as JSON/table
                    st.dataframe(results, use_container_width=True)
            except Exception as e:
                st.dataframe(results, use_container_width=True)

            # Show detailed descriptions
            if show_details:
                with st.expander("Detailed Descriptions"):
                    for item in results:
                        st.markdown(f"### {item.get('name', 'N/A')}")
                        st.markdown(f"**[Open Assessment]({item.get('url', '#')})**")
                        
                        # Show score if available
                        if "score" in item:
                            st.write(f"**Score:** {item['score']:.2f}")
                        
                        # Show reason if available (from reranking)
                        if "reason" in item:
                            st.write(f"**Reason:** {item['reason']}")
                        
                        # Show description
                        st.write(item.get("description", "No description provided"))
                        st.divider()

        except requests.exceptions.Timeout:
            st.error("API request timed out. Backend is slow or unreachable.")

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API. Make sure the API is running at {API_URL}")

        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.status_code}")

        except ValueError as e:
            st.error(f"Data validation error: {e}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    st.info(
        """
        **API Configuration:**
        - Set `USE_LOCAL_API=true` env var to use local API
        - Default: Remote API (render.com)
        
        **Features:**
        - Natural language queries
        - Job description matching
        - JD URL support
        - Gemini reranking (local only)
        """
    )
    
    st.header("About")
    st.write("SHL Assessment Recommender System")

if __name__ == "__main__":
    st.run()

