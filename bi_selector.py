import pandas as pd
from itertools import combinations
import google.generativeai as genai
from scipy.stats import pearsonr, chi2_contingency, ttest_ind, f_oneway
import os
import json
from dotenv import load_dotenv

load_dotenv()
import utils
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY3")

class BivariateSelectorAgent:
    def __init__(self, variable_types: dict, max_pairs: int = 3, correlation_threshold: float = 0.3):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.variable_types = variable_types
        self.max_pairs = max_pairs
        self.correlation_threshold = correlation_threshold

    def compute_statistics(self, df: pd.DataFrame):
        pairs = []

        for var1, var2 in combinations(df.columns, 2):
            type1 = self.variable_types[var1]
            type2 = self.variable_types[var2]

            result = {"pair": [var1, var2], "types": [type1, type2], "test": None, "stat_value": None, "p_value": None}

            # Numerical - Numerical: Pearson Correlation
            if 'numerical' in type1 and 'numerical' in type2:
                try:
                    corr, p_value = pearsonr(df[var1].dropna(), df[var2].dropna())
                    result.update({"test": "Pearson Correlation", "stat_value": corr, "p_value": p_value})
                except:
                    continue

            # Numerical - Categorical: t-test or ANOVA
            elif ('numerical' in type1 and 'categorical' in type2) or ('categorical' in type1 and 'numerical' in type2):
                numerical_var = var1 if 'numerical' in type1 else var2
                categorical_var = var2 if 'numerical' in type1 else var1
                groups = [df[numerical_var][df[categorical_var] == cat].dropna() for cat in df[categorical_var].dropna().unique()]
                try:
                    if len(groups) == 2:
                        stat, p_value = ttest_ind(groups[0], groups[1], equal_var=False)
                        result.update({"test": "t-test", "stat_value": stat, "p_value": p_value})
                    elif len(groups) > 2:
                        stat, p_value = f_oneway(*groups)
                        result.update({"test": "ANOVA", "stat_value": stat, "p_value": p_value})
                except:
                    continue

            # Categorical - Categorical: Chi-square Test
            elif 'categorical' in type1 and 'categorical' in type2:
                try:
                    contingency_table = pd.crosstab(df[var1], df[var2])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    result.update({"test": "Chi-square", "stat_value": chi2, "p_value": p_value})
                except:
                    continue

            if result["test"] is not None:
                pairs.append(result)

        return pairs

    def ask_gemini_to_select_pairs(self, pairs, df, context):
        prompt = f"""
        You are a statistical reasoning assistant. Below are bivariate pairs with their variable types and the statistical test results.
        contex: {context}

        Criteria:
        - If there is anything given in the context about relation of columns, you needto select those columns irrespective of their statitstical results
        - For numerical-numerical pairs: prefer higher absolute correlation.
        - For numerical-categorical pairs: prefer smaller p-values from t-test or ANOVA.
        - For categorical-categorical pairs: prefer smaller p-values from Chi-square test.
        - Select up to {self.max_pairs} most promising bivariate pairs for further analysis.
        - Avoid redundant or weak relationships.
        Provide your selection in JSON format as:
        {{
            "selected_pairs": [
                {{"pair": ["var1", "var2"], "reason": "brief reasoning"}},
                ...
            ]
        }}

        Dataset Summary:
        {df.dtypes.to_dict()}

        Candidate Pairs with Test Results:
        {pairs}
        """

        response = self.model.generate_content(prompt)
        return response.text

    def select_bivariate_pairs(self, file_path: str, context: str):
        df = pd.read_csv(file_path)
        candidate_pairs = self.compute_statistics(df)

        if not candidate_pairs:
            print("No suitable pairs found.")
            return []

        gemini_response = self.ask_gemini_to_select_pairs(candidate_pairs, df, context)
        selected_pairs = json.loads(utils.extract_json_from_response(gemini_response))
        return selected_pairs["selected_pairs"]

# types =  {'Maths': 'numerical discrete', 'Physics': 'numerical discrete', 'Chemistry': 'numerical discrete'}
# bi_selector = BivariateSelectorAgent(types)
# selected_pairs = bi_selector.select_bivariate_pairs("uploads/student_marks_pre.csv", "this is dataset of marks obtained by PU students in PES college and consider maths and physics columns are related")
# print(selected_pairs)
