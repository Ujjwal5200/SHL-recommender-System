# src/app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SHL Recommender", layout="wide")

st.title("SHL Assessment Recommendation System")

st.markdown("""
Enter a natural language query, job description text, or JD URL to get 5–10 most relevant SHL **Individual Test Solutions**.
""")

query = st.text_area("Query / JD text / URL", height=150, placeholder="Example: I am hiring for Java developers who can also collaborate effectively...")

if st.button("Get Recommendations", type="primary"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Finding best assessments..."):
            try:
                response = requests.post("http://127.0.0.1:8000/recommend", json={"query": query}, timeout=30)
                response.raise_for_status()
                data = response.json()["recommended_assessments"]
                
                if data:
                    st.success(f"Found {len(data)} relevant assessments")
                    
                    # Convert to DataFrame for nice table
                    df = pd.DataFrame(data)
                    st.dataframe(
                        df[["name", "url", "test_types", "adaptive_support", "remote_support"]],
                        use_container_width=True,
                        column_config={
                            "url": st.column_config.LinkColumn("URL", display_text="View Details"),
                            "test_types": st.column_config.ListColumn("Test Types")
                        }
                    )
                    
                    # Optional: show descriptions in expander
                    with st.expander("Detailed Descriptions"):
                        for item in data:
                            st.markdown(f"**{item['name']}** ({item['url']})")
                            st.write(item["description"])
                            st.markdown("---")
                else:
                    st.info("No recommendations found – try a more specific query")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to API: {e}")
            except Exception as e:
                st.error(f"Error: {e}")