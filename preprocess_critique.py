import pandas as pd
import google.generativeai as genai
from scipy import stats
import numpy as np
import time
import logging
import warnings
warnings.filterwarnings("ignore")
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class GeminiCodeExecutionCritique:
    def __init__(self, original_csv, preprocessed_csv):
        self.original_df = pd.read_csv(original_csv)
        self.processed_df = pd.read_csv(preprocessed_csv)
        genai.configure(api_key=OOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _clean_code_response(self, response_text):
        code = response_text.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        return code.strip()

    def _validate_generated_code(self, code):
        banned = ['import os', 'import sys', 'exec(', 'eval(', 'open(', '__import__', 'subprocess']
        return not any(word in code for word in banned)

    def generate_and_run_code(self, column, orig_data, proc_data):
        prompt = f"""
        You are a data analyst. Compare whether the original and processed columns come from the same distribution.

        original = {orig_data[:10].tolist()}
        processed = {proc_data[:10].tolist()}
        len_original = {len(orig_data)}, len_processed = {len(proc_data)}

        Requirements:
        1. Use appropriate statistical tests (e.g., t-test, KS test, Mann-Whitney U).
        2. Consider statistical significance and effect size.
        3. Set `distribution_result` to "Same" or "Different".
        4. Set `reason` to a string explaining your reasoning briefly.
        5. Use the variables `original` and `processed` already defined.
        6. Only return valid Python code. No markdown or explanation outside code.
        """

        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                code = self._clean_code_response(response.text)
                if not code or not self._validate_generated_code(code):
                    continue

                safe_builtins = {
                    "abs": abs, "min": min, "max": max, "len": len, "print": print, "__import__": __import__
                }

                exec_env = {
                    "original": orig_data,
                    "processed": proc_data,
                    "stats": stats,
                    "np": np,
                    "__builtins__": safe_builtins
                }

                exec(code, exec_env)

                result = exec_env.get("distribution_result", "No result")
                reason = exec_env.get("reason", "No explanation provided")

                return result, reason
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(0.5)

        return "Error", "Code execution failed"

    def run_comparison(self):
        original_cols = set(self.original_df.columns)
        processed_cols = set(self.processed_df.columns)

        common_cols = list(original_cols & processed_cols)
        missing_cols = list(original_cols - processed_cols)

        results = []

        for col in common_cols:
            orig_col = self.original_df[col].dropna()
            proc_col = self.processed_df[col].dropna()

            if len(orig_col) < 10 or len(proc_col) < 10:
                results.append({
                    "column": col,
                    "analysed": "Insufficient data",
                    "reason": "Not enough data points to perform comparison"
                })
            else:
                result, reason = self.generate_and_run_code(col, orig_col.values, proc_col.values)
                results.append({
                    "column": col,
                    "analysed": result,
                    "reason": reason
                })

        return pd.DataFrame(results), missing_cols


# Example usage
if __name__ == "__main__":
    api_key = "AIzaSyApgtd8xbx5wcYSlcUpWnKkEFfcqvrsX_A"  # 🔐 Replace this with your valid Gemini API key
    checker = GeminiCodeExecutionCritique(
        original_csv="/content/IRIS.csv",
        preprocessed_csv="/content/IRIS_pre.csv",
        api_key=api_key
    )

    result_df, missing_columns = checker.run_comparison()

    print("\n🔍 Final Analysis:\n")
    for _, row in result_df.iterrows():
        print(f"📊 Column: {row['column']}")
        print(f"   ➤ Analysed: {row['analysed']}")
        print(f"   💬 Reason  : {row['reason']}\n")

    print("\n🧾 Summary Table:\n")
    print(result_df.to_string(index=False))

    if missing_columns:
        print(f"\n⚠️ Unnecessary columns skipped: {set(missing_columns)}")
    else:
        print("\n✅ All columns in the original file were present in the preprocessed file.")
