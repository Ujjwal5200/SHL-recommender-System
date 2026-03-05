# src/app.py
import streamlit as st
import requests
import pandas as pd
from typing import List, Dict

# ---------------- CONFIG ----------------
import os
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/recommend")
REQUEST_TIMEOUT = 30

st.set_page_config(
    page_title="SHL Recommender",
    layout="wide",
    page_icon="📊"
)

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def fetch_recommendations(query: str) -> List[Dict]:
    response = requests.post(
        API_URL,
        json={"query": query},
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    payload = response.json()

    if "recommended_assessments" not in payload:
        raise ValueError("Invalid API response structure")

    return payload["recommended_assessments"]

def validate_data(data: List[Dict]) -> pd.DataFrame:
    required_cols = {
        "name", "url", "test_types",
        "adaptive_support", "remote_support"
    }

    df = pd.DataFrame(data)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing fields in API data: {missing}")

    return df[list(required_cols)]

# ---------------- UI ----------------
st.title("SHL  Recommendation System")

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
        st.stop()

    with st.spinner("Finding best assessments..."):
        try:
            results = fetch_recommendations(query)

            if not results:
                st.info("No recommendations found. Try a more specific query.")
                st.stop()

            df = validate_data(results)

            st.success(f"Found {len(df)} relevant assessments")

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

            with st.expander("Detailed Descriptions"):
                for item in results:
                    st.markdown(f"### {item['name']}")
                    st.markdown(f"[Open Assessment]({item['url']})")
                    st.write(item.get("description", "No description provided"))
                    st.divider()

        except requests.exceptions.Timeout:
            st.error("API request timed out. Backend is slow or unreachable.")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to recommendation API.")

        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.status_code}")

        except ValueError as e:
            st.error(f"Data validation error: {e}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")