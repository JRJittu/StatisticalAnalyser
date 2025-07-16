import streamlit as st
import pandas as pd
import os
from core_agent import CoreAgent
from query_agent import QueryAgent
import json

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Automated Statistical Analysis using LLM")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    upload_file_name = uploaded_file.name
    file_path = os.path.join(UPLOAD_DIR, upload_file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    data_context = st.text_input("Provide a data context or description of the dataset (optional):")

    if st.button("Run Analysis"):
        try:
            st.write("Analyzing the CSV with Core Agent...")
            core_agent = CoreAgent()

            if not data_context.strip():
                data_context = "General statistical analysis"

            combined_result_file, selected_columns, selected_pairs = core_agent.analyse_dataset(file_path, upload_file_name, data_context)
            st.success("Analysis completed!")
            st.session_state['combined_result_file'] = combined_result_file
            st.session_state['selected_columns'] = selected_columns
            st.session_state['selected_pairs'] = selected_pairs
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
           
if 'combined_result_file' in st.session_state:
    with open(st.session_state['combined_result_file'], "r", encoding="utf-8") as f:
        result_json = json.load(f)

    selected_columns = st.session_state.get('selected_columns', {})
    selected_pairs = st.session_state.get('selected_pairs', [])

    st.markdown("### 📊 Analysis Summary")

    with st.expander("🧹 Preprocessing Results"):
        st.subheader("Outlier Detection")
        st.json(result_json["preprocessing"]["outlier_result"])
        st.subheader("Distribution Comparison")
        st.json(result_json["preprocessing"]["distribution_result"])

    with st.expander("📈 Univariate Analysis"):
        univariate = result_json.get("univariate", {})
        desc_results = univariate.get("descriptive", {})
        visual_results = univariate.get("visual", {})
        infer_results = univariate.get("inferential", {})

        for col, col_type in selected_columns.items():
            if col in desc_results or col in visual_results or col in infer_results:
                with st.expander(f"📌 Column: {col}"):
                    # Descriptive Stats
                    if col in desc_results:
                        st.subheader("Descriptive Statistics")
                        st.json(desc_results[col])

                    # Visualizations
                    if col in visual_results:
                        st.subheader("Visualizations")
                        # st.json(visual_results[col])
                        visuals = visual_results[col]
                        for vis_key, vis_info in visuals.items():
                            if isinstance(vis_info, dict):
                                image_path = f"uploads/{col}_vis{vis_key[-1]}.png"
                                if os.path.exists(image_path):
                                    st.image(image_path, caption=f"{col} - {vis_info.get('name', vis_key)}")

                    # Inferential Stats
                    if col in infer_results:
                        st.subheader("Inferential Statistics")
                        st.json(infer_results[col])

    with st.expander("📉 Bivariate Analysis"):
        bivariate = result_json.get("bivariate", {})
        bi_desc = bivariate.get("descriptive", {})
        bi_vis = bivariate.get("visual", {})
        bi_inf = bivariate.get("inferential", {})

        for pair_obj in selected_pairs:
            col1, col2 = pair_obj["pair"]
            pair_key = f"{col1}-{col2}"

            if pair_key in bi_desc or pair_key in bi_vis or pair_key in bi_inf:
                with st.expander(f"🔗 Pair: {col1} vs {col2}"):
                    # Descriptive Stats
                    if pair_key in bi_desc:
                        st.subheader("Descriptive Statistics")
                        st.json(bi_desc[pair_key])

                    # Visualizations
                    if pair_key in bi_vis:
                        st.subheader("Visualizations")
                        # st.json(bi_vis[pair_key])
                        visuals = bi_vis[pair_key]
                        for vis_key, vis_info in visuals.items():
                            if isinstance(vis_info, dict):
                                image_path = f"uploads/bi_{col1}_{col2}_vis{vis_key[-1]}.png"
                                if os.path.exists(image_path):
                                    st.image(image_path, caption=f"{pair_key} - {vis_info.get('name', vis_key)}")

                    # Inferential Stats
                    if pair_key in bi_inf:
                        st.subheader("Inferential Statistics")
                        st.json(bi_inf[pair_key])

# ================= QUERY SECTION =================
    st.write("You can now query the analysis results:")
    query = st.text_input("Enter your query:")

    if query:
        try:
            query_agent = QueryAgent(file_path=st.session_state['combined_result_file'])
            answer = query_agent.get_answer(query)
            st.markdown("### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error answering query: {str(e)}")
