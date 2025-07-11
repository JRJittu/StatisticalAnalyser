import pandas as pd
import numpy as np
import json
import google.generativeai as genai
from typing import Dict, List, Any
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy import stats
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
from kb_statistical import StatisticalKnowledgeBase
from utils import util_functions

class BivariateAnalyzer:
    def __init__(self, knowledge_base: StatisticalKnowledgeBase):
        self.knowledge_base = knowledge_base
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.knowledge = None
        self.priority_test_result = None

    def fetch_knowledge(self, var_type):
        doc = self.knowledge_base.search_knowledge("bivariate", var_type)
        if doc == "No relevant statistical test found.":
            raise ValueError("No statistical knowledge found for this variable type.")
        self.knowledge = json.loads(doc)

    def analyze(self, data_column1: pd.Series, var_type1: str, col_name1: str, metadata1: str, data_column2: pd.Series, var_type2: str, col_name2: str, metadata2: str):
        combined_var_type = f"{var_type1} + {var_type2}"
        self.fetch_knowledge(combined_var_type)

        self.desc_result = self.perform_descriptive_stats(data_column1, metadata1, data_column2, metadata2)
        self.perform_visualization(data_column1, col_name1, data_column2, col_name2, self.desc_result)
        inferential_result = self.perform_inferential_stats(data_column1, metadata1, data_column2, metadata2, self.desc_result)

        # return {
            # "descriptive": desc_result,
            # "visualization": self.visualization_suggestions,
            # "inferential": inferential_result,
            # "priority_tests": self.priority_test_data
        # }

    def perform_descriptive_stats(self, data_column1: pd.Series, metadata1: str, data_column2: pd.Series, metadata2: str):
        priority_tests = self.knowledge.get("priority_tests", [])
        descriptive_stats = self.knowledge.get("descriptive", {}).get("statistics", [])
        selection_criteria = self.knowledge.get("descriptive", {}).get("selection_criteria", [])
        application_criteria = self.knowledge.get("descriptive", {}).get("application_criteria", [])

        descriptive_prompt = f"""
            You are a statistical Python code generator. Generate Python code to calculate the following using numpy, pandas, scipy and statsmodels as needed.

            Priority Tests:
            {json.dumps(priority_tests, indent=2)}

            Descriptive Statistics:
            {json.dumps(descriptive_stats, indent=2)}

            Instructions:
            - Assume 'data_column1' and 'data_column2' are pandas Series already defined and use them as variables in code you generates.
            - Calculate all mentioned priority tests and descriptive statistics based on the two variables.
            - The python code you generates, should store the results in a dictionary named 'result' in the following format:
                {{
                    "priority": {{
                        "priority_test_1": result_value,
                        "priority_test_2": result_value,
                        ...
                    }},
                    "descriptive": {{
                        "statistic_1": result_value,
                        "statistic_2": result_value,
                        ...
                    }}
                }}
            - Return ONLY the Python code string that performs the calculations.
            - Do not include any comments, explanations, or additional text.
            - Do not define any functions.
            - Don't include any print statements in code string, and only return executable string without any error.
        """

        response = self.model.generate_content(descriptive_prompt)
        python_code = util_functions.extract_json_from_response(response.text)

        local_vars = {
            'data_column1': data_column1,
            'data_column2': data_column2
        }
        exec(python_code, {}, local_vars)

        intermediate_result = local_vars.get('result')
        serializable_result = util_functions.convert_to_serializable(intermediate_result)
        print("serial: ", serializable_result)

        reasoning_prompt = f"""
            You are a statistical reasoning assistant. Based on the following bivariate test results, selection criteria, and application criteria, finalize the preferred statistics and summarize key findings.

            Test Results:
            {json.dumps(serializable_result, indent=2)}

            Selection Criteria:
            {json.dumps(selection_criteria, indent=2)}

            Application Criteria:
            {json.dumps(application_criteria, indent=2)}

            Metadata:
            Column 1: {metadata1}
            Column 2: {metadata2}

            Instructions:
            - For all statistics test provided with result, selection criteria, and application criteria decide on why that test is better or not.
            - Return ONLY a JSON dictionary in this format:
                {{
                    "statistics_results": {{
                        "selected_stat_1": {{
                            "result_value": value or "NA",
                            "result_text": "result in simple human understandable terms",
                            "preferred": "boolean True or False",
                            "reason": "reason for why this statistic preferred/not preferred"
                        }},
                        "selected_stat_2": {{
                            "result_value": value or "NA",
                            "result_text": "result in simple human understandable terms",
                            "preferred": "boolean True or False",
                            "reason": "reason for why this statistic preferred/not preferred"
                        }},
                        ...
                    }}
                }}
            - Do not return any additional text or explanation outside the JSON structure.
        """

        response = self.model.generate_content(reasoning_prompt)
        json_string = util_functions.extract_json_from_response(response.text)

        descriptive_result = json.loads(json_string)
        print("\nFinal Bivariate Descriptive Statistics Result:")
        for k, v in descriptive_result["statistics_results"].items():
            print(k, " : ", v)

        return descriptive_result

    def perform_visualization(self, data_column1, column_name1, data_column2, column_name2, desc_result):
        prompt = f"""
            You are a statistical visualization assistant. Based on statistical knowledge and the provided **two** data columns and descriptive statistics, suggest the **two most appropriate bivariate visualizations**.

            Visualization Options and Selection Criteria:
            {json.dumps(self.knowledge.get("visualization", {}), indent=2)}

            Descriptive Statistics Result:
            {json.dumps(desc_result, indent=2)}

            Data Column 1 - {column_name1} describe:
            {data_column1.describe()}

            Data Column 2 - {column_name2} describe:
            {data_column2.describe()}

            Instructions:
            - Assume 'data_column1' and 'data_column2' are pandas Series already defined.
            - Select the two most appropriate visualizations based on the data types, distribution, and descriptive statistics.
            - For each visualization, provide:
                - "name": the name of the selected plot (e.g., 'Scatter Plot', 'Hexbin Plot', 'Box Plot by Category').
                - Python code using matplotlib.pyplot to visualize the relationship.
                - Use the variables `data_column1` and `data_column2` in the code directly.
                - Do not include any comments and print statements.
                - Save each plot as: '{column_name1}_{column_name2}_vis1.png' and 'bivariate_{column_name1}_{column_name2}_vis2.png'.
                - Choose graph size and axis ranges based on the min and max values of the data.
                - "reason": clearly explain why this plot is suitable for the given pair of columns.
                - If only one visualization is applicable, return just one.
            - Return ONLY a JSON dictionary in the following format:
                {{
                    "visualization_1": {{
                        "name": "name of selected plot",
                        "python_code": "code_here",
                        "reason": "reason_here"
                    }},
                    "visualization_2": {{
                        "name": "name of selected plot",
                        "python_code": "code_here",
                        "reason": "reason_here"
                    }}
                }}
        """

        print("\nBivariate Visualization Suggestions:")
        response = self.model.generate_content(prompt)

        json_string = util_functions.extract_json_from_response(response.text)
        visualization_suggestions = json.loads(json_string)

        for k, v in visualization_suggestions.items():
            print(k, " : ", v)

        local_vars = {
            'plt': plt,
            'data_column1': data_column1,
            'data_column2': data_column2
        }

        if "visualization_1" in visualization_suggestions:
            exec_code_1 = visualization_suggestions['visualization_1']['python_code']
            exec(exec_code_1, local_vars)

        if "visualization_2" in visualization_suggestions:
            exec_code_2 = visualization_suggestions['visualization_2']['python_code']
            exec(exec_code_2, local_vars)


    def perform_inferential_stats(self, data_column1, metadata1, data_column2, metadata2, desc_result):
        inferential_tests = self.knowledge.get("inferential", {}).get("tests", [])
        selection_criteria = self.knowledge.get("inferential", {}).get("selection_criteria", [])
        application_criteria = self.knowledge.get("inferential", {}).get("application_criteria", [])

        inferential_prompt = f"""
            You are a statistical inference expert. Perform the following task carefully:
            Available Inferential Tests:
            {json.dumps(inferential_tests, indent=2)}
            Selection Criteria:
            {json.dumps(selection_criteria, indent=2)}
            Application Criteria:
            {json.dumps(application_criteria, indent=2)}
            Descriptive Statistics Results:
            {json.dumps(desc_result, indent=2)}
            Metadata for Variable 1:
            {metadata1}
            Metadata for Variable 2:
            {metadata2}

            Instructions:
            - Select appropriate **bivariate** inferential statistical tests from the provided list based on the selection criteria, application criteria, and descriptive statistics.
            - For each selected test, write the Null Hypothesis (H₀) and the Alternative Hypothesis (H₁) if applicable using the provided metadata.
            - DO NOT perform the statistical test. Instead, return Python code that, when executed, will perform the test using 'data_column1' and 'data_column2', which are already defined.
            - The Python code should store the test result in a dictionary named 'result' following this structure:
                {{
                    "test_statistic": calculated_value,
                    other values of that statistics
                }}
            - Do not defined any functions, do not include print statements, or comments in the Python code.
            - Return ONLY a JSON dictionary in this EXACT format:
                {{
                    "inf_test_1_name": {{
                        "hypothesis": "write hypothesis here including H₀ and H₁ if required",
                        "python_code": "write Python code as a string here that produces the 'result' dictionary",
                        "reason": "why this test was selected"
                    }},
                    "inf_test_2_name": {{
                        "hypothesis": "write hypothesis here including H₀ and H₁ if required",
                        "python_code": "write Python code as a string here that produces the 'result' dictionary",
                        "reason": "why this test was selected"
                    }}
                }}
            - Do not return any explanations or text outside this JSON structure.
        """

        response = self.model.generate_content(inferential_prompt)
        json_string = util_functions.extract_json_from_response(response.text)
        inferential_results = json.loads(json_string)

        for test_name, test_details in inferential_results.items():
            local_vars = {'data_column1': data_column1, 'data_column2': data_column2}
            print(f"{test_name} pythong code: ", test_details['python_code'])
            exec(test_details['python_code'], local_vars)
            result = local_vars.get('result')

            print(f"\nExecuted Result for {test_name}: {result}")
            inferential_results[test_name]['result'] = result

            del inferential_results[test_name]['python_code']
        print("\n\nCompleted Execution\n\n")
        conclusion_prompt = f"""
            You are a statistical inference reasoning assistant.

            Test Selection and results:
            {json.dumps(inferential_results, indent=2)}

            Instructions:
            - Based on the executed results and the initial test selection, provide the final conclusion for each test.
            - For each test, write the conclusion in simple human-understandable language.
            - Return the updated JSON with an additional key 'conclusion' added for each test.
            - The returned format should look like this:
                {{
                    "inf_test_1_name": {{
                        "hypothesis": "as provided",
                        "reason": "as provided",
                        "result": "as provided",
                        "conclusion": "final conclusion based on executed result"
                    }},
                    "inf_test_2_name": {{
                        "hypothesis": "as provided",
                        "reason": "as provided",
                        "result": "as provided",
                        "conclusion": "final conclusion based on executed result"
                    }}
                }}
            - Do not return any explanations or text outside this JSON structure.
        """

        conclusion_response = self.model.generate_content(conclusion_prompt)
        final_json_string = util_functions.extract_json_from_response(conclusion_response.text)
        final_inferential_results = json.loads(final_json_string)

        print("\nFinal Inferential Statistics Results:")
        for k, v in final_inferential_results.items():
            print(k, " : ", v)

        return final_inferential_results
