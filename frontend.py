import streamlit as st
import pandas as pd
import os
from core_agent import CoreAgent
from query_agent import QueryAgent

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

            combined_result_file = core_agent.analyse_dataset(file_path, upload_file_name, data_context)
            st.success("Analysis completed!")
            st.session_state['combined_result_file'] = combined_result_file

        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")

if 'combined_result_file' in st.session_state:
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
