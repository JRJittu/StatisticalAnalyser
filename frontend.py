import streamlit as st
import pandas as pd
import os
from core_agent import CoreAgent

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

    # try:
    #     st.write("Analyzing the CSV with Core Agent...")
    #     analysis_result = CoreAgent.analyse_dataset(file_path)

    #     st.subheader("Summary Statistics")
    #     st.json(analysis_result["summary_statistics"])

    #     st.subheader("Missing Values")
    #     st.json(analysis_result["missing_values"])

    #     st.info(f"Total Rows: {analysis_result['row_count']}")
    #     st.info(f"Total Columns: {analysis_result['column_count']}")

    # except Exception as e:
    #     st.error(f"Error analyzing file: {str(e)}")
