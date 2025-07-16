import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
from uni_agent import UnivariateAnalyzer
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY2")

class UniCritique:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.uni_agent = UnivariateAnalyzer(self.knowledge_base)
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def get_knowledge_for_variable(self, var_type: str) -> Dict:
        doc = self.knowledge_base.search_knowledge("univariate", var_type)
        if doc == "No relevant statistical test found.":
            raise ValueError(f"No statistical knowledge found for variable type: {var_type}")
        return json.loads(doc)

    def validate(self, data_column: pd.Series, var_type: str, metadata: str, column_name: str,  desc_results, visual_result, infer_result):
        self.knowledge = self.get_knowledge_for_variable(var_type)
        self.desc_result = desc_results
        self.visual_result = visual_result
        self.infer_result = infer_result
        self.metadata = metadata
        self.data_column = data_column
        self.column_name = column_name
        self.uni_agent.fetch_knowledge(var_type)

        self.validate_descriptive_statistics()
        self.validate_visualizations()
        self.validate_inferential_statistics()

        return self.desc_result, self.visual_result, self.infer_result


    def validate_descriptive_statistics(self):
        try:
            priority_tests = self.knowledge.get("priority_tests", [])
            descriptive_stats = self.knowledge.get("descriptive", {})
            retries = 3

            for i in range(retries):
                print("\nDescriptive statistics retry: ", i+1) 
                validation_prompt = f"""
                You are a statistical validation expert. Validate whether the calculated statistics match the knowledge base requirements.

                Knowledge Base Requirements:
                Priority Tests: {json.dumps(priority_tests, indent=2)}
                Descriptive statistics along with application criteria and selection criteria:
                {json.dumps(descriptive_stats, indent=2)}

                Selected descriptive tests and their results:
                {json.dumps(self.desc_result, indent=2)}

                Metadata:
                {self.metadata}

                Instructions:
                - Check if all required priority tests from knowledge base are calculated
                - Check if appropriate descriptive statistics from knowledge base are calculated
                - Verify if the calculations are appropriate for the data characteristics
                - Also validate the codes if any in the descriptive statistics result are correct. If code contains any variables such as `data_column` then don't consider it an error
                - Return only string "TRUE" if all results are correct
                - If there is any error such as code error or incorrect statistical method chosen, return ONLY that error string
                - Do not return a response of more than 100 words
                """

                response = self.model.generate_content(validation_prompt)
                validation_feedback = response.text.strip()
                print("validation_feedback:", validation_feedback)
                if "TRUE" in validation_feedback.upper():
                    # print("Univariate Descriptive Result Validated")
                    return
                else:
                    # print("Univariate Descriptive Result Not Validated")
                    self.desc_result = self.uni_agent.perform_descriptive_stats(self.data_column, self.metadata, validation_feedback)

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_visualizations(self):
        try:
            visualizations = self.knowledge.get("visualization", {})
            retries = 3

            for attempt in range(retries):
                print("\nVisualization validation retry:", attempt + 1)
                validation_prompt = f"""
                You are a visualization validation expert. Validate whether the selected visualizations are appropriate.

                Knowledge Base Visualization options along with their selection criteria:
                {json.dumps(visualizations, indent=2)}

                Selected Visualizations:
                {json.dumps(self.visual_result, indent=2)}

                Descriptive Statistics Results:
                {json.dumps(self.desc_result, indent=2)}

                Metadata:
                {self.metadata}

                Instructions:
                - Check if the selected visualizations are part of the knowledge base options
                - Verify if selections are appropriate based on data characteristics and descriptive statistics
                - If the selection contains more than two visualizations, flag it as an error. If one or two are selected, it's acceptable.
                - Consider sample size, distribution shape, presence of outliers, etc.
                - Return only the string "TRUE" if the selected visualizations are correct
                - If there is a mistake in the selected visualization methods, return ONLY that error string
                - Do not return a response longer than 100 words
                """
                response = self.model.generate_content(validation_prompt)
                validation_feedback = response.text.strip()
                print("validation_feedback:", validation_feedback)
                if "TRUE" in validation_feedback.upper():
                    # print("Univariate Visualization Result Validated")
                    return
                else:
                    # print("Univariate Visualization Result Not Validated")
                    self.visual_result = self.uni_agent.perform_visualization(self.desc_result, self.column_name, validation_feedback)

        except Exception as e:
            return False, f"Visualization validation error: {str(e)}"


    def validate_inferential_statistics(self):
        try:
            inferential_options = self.knowledge.get("inferential", {})
            available_tests = inferential_options.get("tests", [])
            selection_criteria = inferential_options.get("selection_criteria", [])
            application_criteria = inferential_options.get("application_criteria", [])
            retries = 3

            for i in range(retries):
                print("\nInferential statistics retry: ", i+1)
                validation_prompt = f"""
                You are an inferential statistics validation expert. Validate whether the selected inferential tests for the given column are appropriate using the knowledge provided.

                Knowledge Base Inferential Tests:
                Available Tests: {json.dumps(available_tests, indent=2)}
                Selection Criteria: {json.dumps(selection_criteria, indent=2)}
                Application Criteria: {json.dumps(application_criteria, indent=2)}

                Selected inferential tests and their results on the given column:
                {json.dumps(self.infer_result, indent=2)}

                Descriptive Statistics Results for selecting inferential tests:
                {json.dumps(self.desc_result, indent=2)}

                Metadata:
                {self.metadata}

                Instructions:
                - Check if the selected tests are part of the knowledge base
                - Verify if test selections meet the selection criteria
                - Consider data characteristics (normality, sample size, etc.) from descriptive results
                - Only validate the **results of inferential statistics**, not recomputation
                - Also validate any code snippets within the inferential results; if code includes variables like `data_column`, ignore as error
                - Check if hypotheses are properly formulated given the metadata
                - Return only the string "TRUE" if all results are correct
                - If there is an error (code, methodology, test selection, etc.), return ONLY that error string
                - Do not return a response longer than 100 words
                """

                response = self.model.generate_content(validation_prompt)
                validation_feedback = response.text.strip()

                if "TRUE" in validation_feedback.upper():
                    # print("\nUnivariate Inferential Result Validated")
                    return
                else:
                    # print("Univariate Inferential Result Not Validated")
                    self.infer_result = self.uni_agent.perform_inferential_stats(self.data_column, self.desc_result, self.metadata, validation_feedback)

        except Exception as e:
            return False, f"Inferential validation error: {str(e)}"
