import json
import google.generativeai as genai
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy as scipy
import os

import utils
from kb_preprocess import PreprocessorKB
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1")

class PreprocessorAgent:
    def __init__(self, knowledge_base: PreprocessorKB):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.knowledge_base = knowledge_base
        self.preprocess_knowledge = None
        self.prior_test_res = None
        self.outlier_result = None
        self.missing_value_result = None

    def fetch_knowledge(self, var_type):
        temp = self.knowledge_base.search_knowledge(var_type)
        self.preprocess_knowledge = json.loads(temp)
        return

    def metadata_generator(self, column_types: dict, context: str = None):
        prompt = f"""
            You are meta data generator expert. for given set of column and along with data types, generate metadata of each column.
            {column_types}
            {context}

            return the output in JSON format.
            {{
                column_name: its_description_string,
                column_name: its_description_string,
                ....,
                dataset_desc: small description of entire dataset.
            }}
        """

        response = self.model.generate_content(prompt)
        metadata = utils.extract_json_from_response(response.text)
        try:
            metadata1 = json.loads(metadata)
            return metadata1
        except Exception as e:
            print("Error parsing LLM response:", e)
            print("LLM raw output:\n", response.text)
            return {}

    def outlier_detector(self, data_column, data_type, metadata):
        prior_tests = self.preprocess_knowledge.get('prior_tests', [])
        if not prior_tests:
            return {"error": "No prior tests found in knowledge base."}

        prompt_1 = f"""
            You are a data scientist. For the following statistical tests, generate executable Python code that directly uses a variable called `data_column`:

            Statistical Tests:
            {json.dumps(prior_tests, indent=2)}

            Instructions:
            - DO NOT define any functions.
            - DO NOT include any import statements.
            - Assume `data_column` is already defined and contains the numerical data.
            - Write executable Python code that runs the tests directly and stores the results.
            - Ensure that code should be executable, it shouldn't have any errors.
            - Convert all NumPy or SciPy results into plain Python types using float() or int() if needed.
            - Create a dictionary named `results` that contains the test results in this format:
            {{
              "test_name_1": result_1,
              "test_name_2": result_2,
              ...
            }}
            - The final line of your code must define this `results` dictionary.
            - Return ONLY the Python code. No comments, no markdown, and no explanations.
        """

        response = self.model.generate_content(prompt_1)
        prior_code = utils.extract_json_from_response(response.text)
        local_vars = {}
        try:
            exec(prior_code, {"np": np, "pd": pd, "stats": stats, "scipy": scipy, "data_column": data_column}, local_vars)
        except Exception as e:
            print(f"[ERROR] Failed to execute prior test code: {e}")
            return {"error": "Execution of prior test code failed."}

        raw_results = local_vars.get("results", {})
        self.prior_test_res = utils.convert_to_serializable(raw_results)

        outlier_methods = self.preprocess_knowledge.get('outlier_detection', [])
        if not outlier_methods:
            return {"error": "No outlier detection methods found in knowledge base."}

        prompt_2 = f"""
            You are a data preprocessing expert. Choose the best outlier detection method based on the following:

            - Data type: {data_type}
            - Column metadata: {metadata}
            - Prior test results: {json.dumps(self.prior_test_res, indent=2)}
            - Available outlier detection methods: {json.dumps(outlier_methods, indent=2)}

            Instructions:
            - Select the most suitable method based on prior test results and data characteristics.
            - Return your selection and reasoning in this JSON format:
            {{
              "selected_method": "Name of selected method",
              "reasoning": "Why this method fits the data",
              "python_code": "Python code to apply this outlier detection method on `data_column`"
            }}
            - Only return the JSON. Do not include markdown or extra explanations.
            - In the python code, DO NOT define any functions.
            - DO NOT include any import statements.
            - In the python code, store the detected outlier indexes in the variable name `outlier_indexes`.
            - Write executable Python code that runs the tests directly and stores the results.
            - Convert all NumPy or SciPy results into plain Python types using float() or int() if needed.
        """
        response2 = self.model.generate_content(prompt_2)
        method_response = utils.extract_json_from_response(response2.text)
        selected_method_json = json.loads(method_response)
        outlier_vars = {}
        try:
            exec(selected_method_json["python_code"], {"np": np, "pd": pd, "stats": stats, "scipy":scipy, "data_column": data_column}, outlier_vars)
        except Exception as e:
            print(f"[ERROR] Failed to execute outlier detection code: {e}")
            return {
                "selected_method": selected_method_json.get("selected_method"),
                "reasoning": selected_method_json.get("reasoning"),
                "error": "Execution of selected outlier detection code failed."
            }

        self.outlier_result = {
            "selected_method": selected_method_json.get("selected_method"),
            "reasoning": selected_method_json.get("reasoning"),
            "outlier_indexes": utils.convert_to_serializable(outlier_vars.get('outlier_indexes', []))
        }

        return self.outlier_result

    def missing_value_imputer(self, data_column, data_type, metadata):
        import builtins

        missing_value_methods = self.preprocess_knowledge.get('missing_value_imputation', [])
        if not missing_value_methods:
            return {"error": "No missing value imputation methods found in knowledge base."}

        prompt = f"""
            You are a data expert. You are given with missing value imputation methods and some results applied on data column.
            - Select appropriate missing value method based on given results, metadata and column type.
            missing_value_method: {missing_value_methods}
            metadata: {metadata}
            column_type: {data_type}
            outlier_detected: {self.outlier_result}

            Instructions:
            - Select the most suitable method based on prior test results and data characteristics.
            - You also need to return the python code to apply the selected method on `data_column`.
            - Assume data_column is already defined and contains data.
            - Write executable Python code that runs the imputation directly and stores the results.
            - Convert all NumPy or SciPy results into plain Python types using float() or int() if needed.
            - Do not include any import statements.
            - DO NOT define any functions.
            - Return your selection and reasoning in this JSON format:
            {{
              "selected_method": "Name of selected method",
              "reasoning": "Reason why this method selected",
              "python_code": "Python code to apply this missing value imputation method on `data_column`"
            }}
        """

        response = self.model.generate_content(prompt)
        response_json = json.loads(utils.extract_json_from_response(response.text))
        local_vars = {"data_column": data_column}
        try:
            exec(response_json["python_code"], {}, local_vars)
            imputed_column = local_vars.get("data_column", data_column)
            self.missing_value_result = {
                "selected_method": response_json["selected_method"],
                "reasoning": response_json["reasoning"],
                "imputed_data": imputed_column
            }
            return self.missing_value_result
        except Exception as e:
            return {
                "error": f"Failed to execute imputation code: {str(e)}",
                "selected_method": response_json.get("selected_method"),
                "reasoning": response_json.get("reasoning"),
                "code": response_json.get("python_code")
            }

    def feature_remover(self, column_data_type: dict, metadata: dict, context: str = None):
        prompt = f"""
            You are a feature selection expert helping with preprocessing of tabular data.

            Task:
            - Analyze the dataset's column names, their data types, and metadata.
            - Remove features that are irrelevant, redundant, or not useful for modeling.
            - Provide an updated dictionary of column names with their types after removing the unnecessary features.
            - Remove date and time related columns if any

            Input:
            Column Data Types: {json.dumps(column_data_type, indent=2)}
            Metadata: {json.dumps(metadata, indent=2)}
            Context: {context if context else "None"}

            - Return a clean Python dictionary of the remaining column names and types.
            - Return the result in this format:
            {{
              "column_name_1": "type_1",
              "column_name_2": "type_2",
              ...
            }}
            - Do NOT include markdown, explanations, or extra text.
        """

        response = self.model.generate_content(prompt)
        cleaned_json_text = utils.extract_json_from_response(response.text)
        try:
            cleaned_column_data_type = json.loads(cleaned_json_text)
            return cleaned_column_data_type
        except Exception as e:
            print("Error parsing cleaned column data types:", e)
            print("Raw response:\n", response.text)
            return column_data_type
