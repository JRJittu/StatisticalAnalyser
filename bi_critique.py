import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import google.generativeai as genai
import os

from bi_agent import BivariateAnalyzer
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY3")

class BiCritique:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.bi_agent = BivariateAnalyzer(self.knowledge_base, GOOGLE_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def get_knowledge_for_variables(self, var_type1: str, var_type2: str) -> Dict:
        """Get knowledge base recommendations for bivariate variable types"""
        combined_var_type = f"{var_type1} + {var_type2}"
        doc = self.knowledge_base.search_knowledge("bivariate", combined_var_type)
        if doc == "No relevant statistical test found.":
            raise ValueError(f"No statistical knowledge found for variable types: {combined_var_type}")
        return json.loads(doc)

    def validate(self, data_column1: pd.Series, var_type1: str, metadata1: str, column_name1: str, 
                 data_column2: pd.Series, var_type2: str, metadata2: str,  column_name2: str,
                 desc_results, visual_result, infer_result):
        
        self.knowledge = self.get_knowledge_for_variables(var_type1, var_type2)
        self.desc_result = desc_results
        self.visual_result = visual_result
        self.infer_result = infer_result
        self.metadata1 = metadata1
        self.metadata2 = metadata2
        self.data_column1 = data_column1
        self.data_column2 = data_column2
        self.column_name1 = column_name1
        self.column_name2 = column_name2
        self.var_type1 = var_type1
        self.var_type2 = var_type2
        self.bi_agent.fetch_knowledge(var_type1, var_type2)

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
                print("\nBivariate Descriptive statistics retry: ", i+1)
                validation_prompt = f"""
                You are a bivariate statistical validation expert. Validate whether the calculated bivariate statistics match the knowledge base requirements.

                Knowledge Base Requirements:
                Priority Tests: {json.dumps(priority_tests, indent=2)}
                Descriptive statistics along with application criteria and selection criteria:
                {json.dumps(descriptive_stats, indent=2)}

                Selected bivariate descriptive tests and their results:
                {json.dumps(self.desc_result, indent=2)}

                Metadata of column1: {self.metadata1}
                Metadata of column2: {self.metadata2}

                Instructions:
                - Check if all required priority tests from knowledge base are calculated for bivariate analysis
                - Check if appropriate descriptive statistics from knowledge base are calculated for bivariate analysis
                - Verify if the calculations are appropriate for the bivariate data characteristics
                - Consider the relationship between the two variables
                - Also validate the codes if any in the descriptive statistics result are correct. If code contains variables such as data_column1, data_column2 then don't consider it an error
                - Return only string "TRUE" if all results are correct
                - If there is any error such as code error or incorrect statistical method chosen, return ONLY that error string
                - Do not return a response of more than 100 words
                """

                response = self.model.generate_content(validation_prompt)
                validation_feedback = response.text.strip()

                if "TRUE" in validation_feedback.upper():
                    print("\nBivariate Descriptive Result Validated")
                    return
                else:
                    print("Bivariate Descriptive Result Not Validated")
                    self.desc_result = self.bi_agent.perform_descriptive_stats(
                        self.data_column1, self.metadata1, self.data_column2, self.metadata2, validation_feedback
                    )

        except Exception as e:
            return False, f"Bivariate descriptive validation error: {str(e)}"

    def validate_visualizations(self):
        try:
            visualizations = self.knowledge.get("visualization", {})
            retries = 3


            for attempt in range(retries):
                print("\nBivariate Visualization validation retry:", attempt + 1)
                validation_prompt = f"""
                You are a bivariate visualization validation expert. Validate whether the selected visualizations are appropriate for bivariate analysis.

                Knowledge Base Visualization options along with their selection criteria:
                {json.dumps(visualizations, indent=2)}

                Selected Bivariate Visualizations:
                {json.dumps(self.visual_result, indent=2)}

                Bivariate Descriptive Statistics Results:
                {json.dumps(self.desc_result, indent=2)}

                Metadata of column1: {self.metadata1}
                Metadata of column2: {self.metadata2}

                Instructions:
                - Check if the selected visualizations are part of the knowledge base options for bivariate analysis
                - Verify if selections are appropriate based on bivariate data characteristics and descriptive statistics
                - Consider the relationship between the two variables
                - If the selection contains more than two visualizations, flag it as an error. If one or two are selected, it's acceptable.
                - Consider sample size, distribution shape, correlation patterns, etc.
                - Return only the string "TRUE" if the selected visualizations are correct
                - If there is a mistake in the selected visualization methods, return ONLY that error string
                - Do not return a response longer than 100 words
                """

                response = self.model.generate_content(validation_prompt)
                validation_feedback = response.text.strip()

                if "TRUE" in validation_feedback.upper():
                    print("\nBivariate Visualization Result Validated")
                    return
                else:
                    print("Bivariate Visualization Result Not Validated")
                    self.visual_result = self.bi_agent.perform_visualization(
                        self.data_column1, self.column_name1, self.data_column2, self.column_name2, self.desc_result, validation_feedback
                    )

        except Exception as e:
            return False, f"Bivariate visualization validation error: {str(e)}"

    def validate_inferential_statistics(self):
        try:
            inferential_options = self.knowledge.get("inferential", {})
            available_tests = inferential_options.get("tests", [])
            selection_criteria = inferential_options.get("selection_criteria", [])
            application_criteria = inferential_options.get("application_criteria", [])
            retries = 3

            for i in range(retries):
                print("\nBivariate Inferential statistics retry: ", i+1)
                validation_prompt = f"""
                You are a bivariate inferential statistics validation expert. Validate whether the selected inferential tests for the given columns are appropriate using the knowledge provided.

                Knowledge Base Inferential Tests:
                Available Tests: {json.dumps(available_tests, indent=2)}
                Selection Criteria: {json.dumps(selection_criteria, indent=2)}
                Application Criteria: {json.dumps(application_criteria, indent=2)}

                Selected bivariate inferential tests and their results on the given columns:
                {json.dumps(self.infer_result, indent=2)}

                Bivariate Descriptive Statistics Results for selecting inferential tests:
                {json.dumps(self.desc_result, indent=2)}

                Metadata:
                Variable 1: {self.metadata1}
                Variable 2: {self.metadata2}

                Instructions:
                - Check if the selected tests are part of the knowledge base for bivariate analysis
                - Verify if test selections meet the selection criteria for the combination of variable types
                - Consider bivariate data characteristics (correlation, independence, etc.) from descriptive results
                - Only validate the results of bivariate inferential statistics, not recomputation
                - Also validate any code snippets within the inferential results; if code includes variables like data_column1, data_column2, ignore as error
                - Check if hypotheses are properly formulated for bivariate relationships given the metadata
                - Return only the string "TRUE" if all results are correct
                - If there is an error (code, methodology, test selection, etc.), return ONLY that error string
                - Do not return a response longer than 100 words
                """

                response = self.model.generate_content(validation_prompt)
                validation_feedback = response.text.strip()

                if "TRUE" in validation_feedback.upper():
                    print("\nBivariate Inferential Result Validated")
                    return
                else:
                    print("Bivariate Inferential Result Not Validated")                    
                    self.infer_result = self.bi_agent.perform_inferential_stats(self.data_column1, self.metadata1, self.data_column2, self.metadata2, self.desc_result, validation_feedback)

        except Exception as e:
            return False, f"Bivariate inferential validation error: {str(e)}"