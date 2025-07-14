import google.generativeai as genai
import pandas as pd
import os
import json
from dotenv import load_dotenv

load_dotenv()

def detect_datatypes(df):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    column_info = {}
    for col in df.columns:
        sample_values = df[col].dropna().head(10).tolist()
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        column_info[col] = {
            'sample_values': sample_values,
            'unique_count': unique_count,
            'total_count': total_count,
            'dtype': str(df[col].dtype)
        }

    prompt = f"""
    Analyze the following CSV column data and classify each column into exactly one of these types:
    - numerical continuous
    - numerical discrete
    - categorical nominal
    - categorical ordinal
    - binary variable
    - time series

    Column Information:
    {column_info}

    For each column, respond with only: "Column_Name: detected_type"
    One line per column. No explanations.
    """

    response = model.generate_content(prompt)

    result_dict = {}
    for line in response.text.strip().splitlines():
        if ':' in line:
            col, dtype = line.split(':', 1)
            result_dict[col.strip()] = dtype.strip()


    if "time series" in result_dict.values():
        for k, v in result_dict.items():
            if v in ["categorical nominal", "categorical ordinal", "binary variable"]:
                continue
            elif v in ["numerical continuous", "numerical discrete"]:
                result_dict[k] = 'time series '+ result_dict[k]

    return result_dict