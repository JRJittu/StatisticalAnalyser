import pandas as pd
import numpy as np
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy import stats

from kb_statistical import StatisticalKnowledgeBase
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class UnivariateAnalyzer:
    def __init__(self, knowledge_base: StatisticalKnowledgeBase):
        self.data = None
        self.var_type = None
        self.knowledge_base = knowledge_base
        self.knowledge = None
        self.priority_test_data = None
        self.metadata = None
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def extract_json_from_response(self, response_text):
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        if "python" in response_text.split('\n')[0]:
            response_text = '\n'.join(response_text.split('\n')[1:])

        return response_text

    def convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def analyze(self, data_column: pd.Series, var_type: str, metadata: str, column_name: str):
        self.data = data_column
        self.var_type = var_type
        self.metadata = metadata
        self.fetch_knowledge(var_type)
        desc_result = self.perform_descriptive_stats(data_column, metadata)
        self.perform_visualization(data_column, desc_result, column_name)
        self.perform_inferential_stats(data_column, desc_result, metadata)

    def fetch_knowledge(self, var_type):
        # print("\nFetching statistical knowledge...")
        doc = self.knowledge_base.search_knowledge("univariate", var_type)
        if doc == "No relevant statistical test found.":
            raise ValueError("No statistical knowledge found for this variable type.")
        self.knowledge = json.loads(doc)

    def perform_descriptive_stats(self, data_column, metadata):
        priority_tests = self.knowledge.get("priority_tests", [])
        descriptive_stats = self.knowledge.get("descriptive", {}).get("statistics", [])
        selection_criteria = self.knowledge.get("descriptive", {}).get("selection_criteria", [])
        application_criteria = self.knowledge.get("descriptive", {}).get("application_criteria", [])

        print("\nDescriptive Statistics Analysis...")

        descriptive_prompt = f"""
        You are a statistical Python code generator. Generate Python code to calculate the following using numpy, pandas, scipy, and statsmodels as needed.

        Priority Tests:
        {json.dumps(priority_tests, indent=2)}

        Descriptive Statistics:
        {json.dumps(descriptive_stats, indent=2)}

        Instructions:
        - Assume 'data_column' is a pandas Series already defined.
        - Calculate all mentioned priority tests and descriptive statistics.
        - Store results in a dictionary named 'result' in the following format:
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
        - Dont include any import statements, any print statements in code string and only return executable string without any error
        """

        response = self.model.generate_content(descriptive_prompt)
        python_code = self.extract_json_from_response(response.text)

        print("\nGenerated Python Code:\n", python_code)

        local_vars = {'data_column': data_column}
        exec(python_code, {}, local_vars)

        intermediate_result = local_vars.get('result')
        serializable_result = self.convert_to_serializable(intermediate_result)
        print("intermediate: ", intermediate_result)
        print("serail: ", serializable_result)

        reasoning_prompt = f"""
        You are a statistical reasoning assistant. Based on the following test results, selection criteria, and application criteria, finalize the preferred statistics and summarize key findings.

        Test Results:
        {json.dumps(serializable_result, indent=2)}

        Selection Criteria:
        {json.dumps(selection_criteria, indent=2)}

        Application Criteria:
        {json.dumps(application_criteria, indent=2)}

        Metadata:
        {metadata}

        Instructions:
        - For all statistics test provided with result, selection criteria, application criteria decide on why that test is better, why not.
        - Return ONLY a JSON dictionary in this format:
            {{
                "statistics_results": {{
                    "selected_stat_1": {{
                        "result_value": value or "NA",
                        "result_text": "result in simple human understandable terms",
                        "preferred": "boolean True or False"
                        "reason": "reason for why this statistic preferred/not preferred",
                    }},
                    "selected_stat_2": {{
                        "result_value": value or "NA",
                        "result_text": "result in simple human understandable terms",
                        "preferred": "boolean True or False"
                        "reason": "reason for why this statistic preferred/not preferred",
                    }},
                    ...
                }}
            }}
        - Do not return any additional text or explanation outside the JSON structure.
        """

        response = self.model.generate_content(reasoning_prompt)
        json_string = self.extract_json_from_response(response.text)

        descriptive_result = json.loads(json_string)
        print("\nFinal Descriptive Statistics Result:")
        for k, v in descriptive_result["statistics_results"].items():
            print(k, " : ", v)

        return descriptive_result


    def perform_visualization(self, data_column, desc_results, column_name):
        prompt = f"""
        You are a statistical visualization assistant. Based on the statistical knowledge and the provided data list and descriptive statistics, suggest the two most appropriate visualizations.

        Visualization Options and Selection Criteria:
        {json.dumps(self.knowledge.get("visualization", {}), indent=2)}

        Descriptive Statistics Result:
        {json.dumps(desc_results, indent=2)}

        Data describe:
        {data_column.describe()}

        column_name: {column_name}

        Instructions:
        - Assume 'data_column' is a pandas Series already defined.
        - Select the two most appropriate visualizations based on the selection criteria and descriptive statistics.
        - For each visualization, provide:
            - "name": the name of the selected plot (e.g., 'Histogram', 'Box Plot').
            - Complete Python code using matplotlib.pyplot.
            - The code no need to include the definition of the data variable at the beginning and use variable name as data_column in the code.
            - Don't include any import statement, comments.
            - The code must save the plot as an image (file name: 'columnName_vis1.png' and 'columnName_vis2.png' respectively).
            - "reason": clearly explain why this visualization is the best choice for the provided data using descriptive statistics.
            - If selecting histogram, use bins size appropriately using range of given data
            - Deside on size and ranges of graph in the code based on max and min values of give data column
        - If not possible to get two best visualization, return only one
        - Return ONLY a JSON dictionary in the following structure:
            {{
                "visualization_1": {{
                    "name": "name of selected plot",
                    "python_code": "code_here",
                    "reason": "reason_here_why you selected that plot"
                }},
                "visualization_2": {{
                    "name": "name of selected plot",
                    "python_code": "code_here",
                    "reason": "reason_here_why you selected that plot"
                }}
            }}
        - Do not return any explanations outside this JSON structure.
        """

        print("\nVisualization Suggestions:")
        response = self.model.generate_content(prompt)
        # print(response.text, "\n")

        json_string = self.extract_json_from_response(response.text)
        visualization_suggestions = json.loads(json_string)
        for k, v in visualization_suggestions.items():
            print(k, " : ", v)

        local_vars = {'plt': plt, 'data_column': data_column}

        if "visualization_1" in visualization_suggestions:
            exec_code_1 = visualization_suggestions['visualization_1']['python_code']
            exec(exec_code_1, local_vars)

        if "visualization_2" in visualization_suggestions:
            exec_code_2 = visualization_suggestions['visualization_2']['python_code']
            exec(exec_code_2, local_vars)

        # return visualization_suggestions


    def perform_inferential_stats(self, data_column, desc_results, metadata):
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
        {json.dumps(desc_results, indent=2)}

        Metadata:
        {metadata}

        Instructions:
        - Select appropriate inferential statistical tests from the provided list based on the selection criteria, application criteria, and descriptive statistics.
        - For each selected test, write the Null Hypothesis (H₀) and the Alternative Hypothesis (H₁) if applicable using the provided metadata.
        - DO NOT perform the statistical test. Instead, return Python code that, when executed, will perform the test using 'data_column' which is already defined.
        - The Python code should store the test result in a dictionary named 'result' following this structure:
            {{
                "test_statistic": calculated_value,
                other values of that statistics
            }}
        - Do not include import statements, print statements, or comments in the Python code.
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
        json_string = self.extract_json_from_response(response.text)
        inferential_results = json.loads(json_string)

        for test_name, test_details in inferential_results.items():
            local_vars = {'data_column': data_column}
            exec(test_details['python_code'], local_vars)
            result = local_vars.get('result')

            print(f"\nExecuted Result for {test_name}: {result}")
            inferential_results[test_name]['result'] = result

            del inferential_results[test_name]['python_code']


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
        final_json_string = self.extract_json_from_response(conclusion_response.text)
        final_inferential_results = json.loads(final_json_string)

        print("\nFinal Inferential Statistics Results:")
        for k, v in final_inferential_results.items():
            print(k, " : ", v)

        return final_inferential_results
