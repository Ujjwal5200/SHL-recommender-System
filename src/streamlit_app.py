# Streamlit App V9 - Production Ready using config_v9.py
import streamlit as st
import requests
from config_v9 import logger

API_URL = "http://localhost:8000/recommend"

st.title("SHL Assessment Recommender V9")

query = st.text_area("Enter query or Job Description:", height=200)

col1, col2 = st.columns([1, 4])
with col1:
    top_k = st.slider("Number of recommendations", 5, 15, 10)
with col2:
    use_rerank = st.checkbox("Use Gemini Reranking", value=True)

if st.button("Get Recommendations"):
    if query:
        try:
            payload = {"query": query, "top_k": top_k, "use_rerank": use_rerank}
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            
            data = resp.json().get("recommended_assessments", [])
            
            if data:
                st.subheader(f"Top {len(data)} Recommendations")
                
                for i, rec in enumerate(data, 1):
                    with st.expander(f"{i}. {rec.get('name', 'N/A')}"):
                        st.write(f"**URL:** {rec.get('url', 'N/A')}")
                        st.write(f"**Score:** {rec.get('score', 0):.2f}")
                        st.write(f"**Reason:** {rec.get('reason', 'N/A')}")
            else:
                st.warning("No recommendations found")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure API is running on port 8000")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")

if __name__ == "__main__":
    st.run()
