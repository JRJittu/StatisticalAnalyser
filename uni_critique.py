import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import google.generativeai as genai
import os

from kb_statistical import StatisticalKnowledgeBase
from utils import util_functions

class UniCritique:
    def __init__(self, knowledge_base: StatisticalKnowledgeBase, GOOGLE_API_KEY):
        self.knowledge_base = knowledge_base
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def get_knowledge_for_variable(self, var_type: str) -> Dict:
        doc = self.knowledge_base.search_knowledge("univariate", var_type)
        if doc == "No relevant statistical test found.":
            raise ValueError(f"No statistical knowledge found for variable type: {var_type}")
        return json.loads(doc)

    def validate_descriptive_statistics(self, data_column: pd.Series, var_type: str, generated_code: str) -> Tuple[bool, str]:
        """
        Validate descriptive statistics against knowledge base recommendations
        """
        try:
            knowledge = self.get_knowledge_for_variable(var_type)

            # Execute the generated code to get results
            local_vars = {'data_column': data_column}
            exec(generated_code, {}, local_vars)

            result = local_vars.get('result')
            if result is None:
                return False, "Generated code did not produce 'result' variable"

            # Convert to serializable format
            serialized_result = self.convert_to_serializable(result)

            # Get knowledge base requirements
            priority_tests = knowledge.get("priority_tests", [])
            descriptive_stats = knowledge.get("descriptive", {}).get("statistics", [])

            validation_prompt = f"""
            You are a statistical validation expert. Validate whether the calculated statistics match the knowledge base requirements.

            Knowledge Base Requirements:
            Priority Tests: {json.dumps(priority_tests, indent=2)}
            Descriptive Statistics: {json.dumps(descriptive_stats, indent=2)}

            Calculated Results:
            {json.dumps(serialized_result, indent=2)}

            Data Characteristics:
            - Sample size: {len(data_column)}
            - Data type: {var_type}
            - Contains nulls: {data_column.isnull().any()}

            Instructions:
            - Check if all required priority tests from knowledge base are calculated
            - Check if all required descriptive statistics from knowledge base are calculated
            - Verify the calculations are appropriate for the data characteristics
            - Return ONLY a JSON with this structure:
            {{
                "validation_passed": true/false,
                "missing_tests": ["list of missing priority tests"],
                "missing_statistics": ["list of missing descriptive statistics"],
                "issues": ["list of any calculation issues"],
                "summary": "brief summary of validation result"
            }}
            """

            response = self.model.generate_content(validation_prompt)
            json_string = util_functions.extract_json_from_response(response.text)
            validation_result = json.loads(json_string)

            return validation_result["validation_passed"], validation_result["summary"]

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_visualizations(self, data_column: pd.Series, var_type: str,
                               desc_results: Dict, selected_visualizations: Dict) -> Tuple[bool, str]:
        """
        Validate visualizations against knowledge base recommendations
        """
        try:
            knowledge = self.get_knowledge_for_variable(var_type)
            visualization_options = knowledge.get("visualization", {})

            validation_prompt = f"""
            You are a visualization validation expert. Validate whether the selected visualizations are appropriate.

            Knowledge Base Visualization Options:
            {json.dumps(visualization_options, indent=2)}

            Selected Visualizations:
            {json.dumps(selected_visualizations, indent=2)}

            Descriptive Statistics Results:
            {json.dumps(desc_results, indent=2)}

            Data Characteristics:
            - Sample size: {len(data_column)}
            - Data range: {data_column.min()} to {data_column.max()}
            - Data type: {var_type}

            Instructions:
            - Check if selected visualizations are available in knowledge base options
            - Verify if selections are appropriate based on data characteristics and descriptive statistics
            - Consider sample size, distribution shape, outliers, etc.
            - Return ONLY a JSON with this structure:
            {{
                "validation_passed": true/false,
                "inappropriate_selections": ["list of inappropriate visualization choices"],
                "better_alternatives": ["list of better visualization options from knowledge base"],
                "reasons": ["reasons why current selections are inappropriate or appropriate"],
                "summary": "brief summary of validation result"
            }}
            """

            response = self.model.generate_content(validation_prompt)
            json_string = util_functions.extract_json_from_response(response.text)
            validation_result = json.loads(json_string)

            return validation_result["validation_passed"], validation_result["summary"]

        except Exception as e:
            return False, f"Visualization validation error: {str(e)}"

    def validate_inferential_statistics(self, data_column: pd.Series, var_type: str,
                                      desc_results: Dict, selected_tests: Dict,
                                      metadata: str) -> Tuple[bool, str]:
        """
        Validate inferential statistics against knowledge base recommendations
        """
        try:
            knowledge = self.get_knowledge_for_variable(var_type)
            inferential_options = knowledge.get("inferential", {})
            available_tests = inferential_options.get("tests", [])
            selection_criteria = inferential_options.get("selection_criteria", [])

            validation_prompt = f"""
            You are an inferential statistics validation expert. Validate whether the selected tests are appropriate.

            Knowledge Base Inferential Tests:
            Available Tests: {json.dumps(available_tests, indent=2)}
            Selection Criteria: {json.dumps(selection_criteria, indent=2)}

            Selected Tests:
            {json.dumps(selected_tests, indent=2)}

            Descriptive Statistics Results:
            {json.dumps(desc_results, indent=2)}

            Metadata: {metadata}

            Data Characteristics:
            - Sample size: {len(data_column)}
            - Data type: {var_type}

            Instructions:
            - Check if selected tests are available in the knowledge base
            - Verify if test selections meet the selection criteria from knowledge base
            - Consider data characteristics (normality, sample size, etc.) from descriptive results
            - Validate that assumptions for selected tests are met
            - Check if hypotheses are properly formulated given the metadata
            - Return ONLY a JSON with this structure:
            {{
                "validation_passed": true/false,
                "inappropriate_tests": ["list of inappropriate test selections"],
                "assumption_violations": ["list of assumption violations"],
                "better_alternatives": ["list of better test options from knowledge base"],
                "hypothesis_issues": ["issues with hypothesis formulation"],
                "summary": "brief summary of validation result"
            }}
            """

            response = self.model.generate_content(validation_prompt)
            json_string = util_functions.extract_json_from_response(response.text)
            validation_result = json.loads(json_string)

            return validation_result["validation_passed"], validation_result["summary"]

        except Exception as e:
            return False, f"Inferential validation error: {str(e)}"

    def comprehensive_validate(self, data_column: pd.Series, var_type: str,
                             generated_code: str, desc_results: Dict,
                             selected_visualizations: Dict, selected_tests: Dict,
                             metadata: str, max_retries: int = 3) -> bool:
        """
        Perform comprehensive validation of the entire analysis
        """
        print("\n" + "="*60)
        print("STARTING KNOWLEDGE BASE VALIDATION")
        print("="*60)

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n>>> RETRY ATTEMPT {attempt} <<<")

            all_passed = True

            # Validate descriptive statistics
            print("\n🔍 VALIDATING DESCRIPTIVE STATISTICS...")
            desc_passed, desc_summary = self.validate_descriptive_statistics(
                data_column, var_type, generated_code
            )

            if desc_passed:
                print("✅ Descriptive statistics validation PASSED")
            else:
                print("❌ Descriptive statistics validation FAILED")
                print(f"   {desc_summary}")
                all_passed = False

            # Validate visualizations
            print("\n🔍 VALIDATING VISUALIZATIONS...")
            viz_passed, viz_summary = self.validate_visualizations(
                data_column, var_type, desc_results, selected_visualizations
            )

            if viz_passed:
                print("✅ Visualization validation PASSED")
            else:
                print("❌ Visualization validation FAILED")
                print(f"   {viz_summary}")
                all_passed = False

            # Validate inferential statistics
            print("\n🔍 VALIDATING INFERENTIAL STATISTICS...")
            inf_passed, inf_summary = self.validate_inferential_statistics(
                data_column, var_type, desc_results, selected_tests, metadata
            )

            if inf_passed:
                print("✅ Inferential statistics validation PASSED")
            else:
                print("❌ Inferential statistics validation FAILED")
                print(f"   {inf_summary}")
                all_passed = False

            if all_passed:
                print("\n✅ All validations PASSED")
                return True
            else:
                if attempt < max_retries - 1:
                    print(f"\n❌ Validation failed. Retrying analysis... (Attempt {attempt + 2}/{max_retries})")
                else:
                    print(f"\n❌ Analysis validation failed after {max_retries} attempts.")

        return False

    def validate_with_recommendations(self, data_column: pd.Series, var_type: str,
                                    analysis_results: Dict) -> Dict[str, Any]:
        """
        Validate analysis and provide specific recommendations for improvement
        """
        try:
            knowledge = self.get_knowledge_for_variable(var_type)

            recommendation_prompt = f"""
            You are a statistical analysis advisor. Provide specific recommendations for improving the analysis.

            Knowledge Base for {var_type}:
            {json.dumps(knowledge, indent=2)}

            Current Analysis Results:
            {json.dumps(analysis_results, indent=2)}

            Data Characteristics:
            - Sample size: {len(data_column)}
            - Data range: {data_column.min()} to {data_column.max()}
            - Has nulls: {data_column.isnull().any()}

            Instructions:
            - Compare current analysis with knowledge base recommendations
            - Identify what's missing or inappropriate
            - Provide specific, actionable recommendations
            - Return ONLY a JSON with this structure:
            {{
                "overall_assessment": "EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR",
                "strengths": ["list of what was done well"],
                "weaknesses": ["list of issues or missing elements"],
                "specific_recommendations": [
                    {{
                        "category": "descriptive/visualization/inferential",
                        "issue": "description of the issue",
                        "recommendation": "specific action to take",
                        "priority": "HIGH/MEDIUM/LOW"
                    }}
                ],
                "summary": "overall summary of the analysis quality"
            }}
            """

            response = self.model.generate_content(recommendation_prompt)
            json_string = util_functions.extract_json_from_response(response.text)
            recommendations = json.loads(json_string)

            return recommendations

        except Exception as e:
            return {
                "overall_assessment": "ERROR",
                "error": str(e),
                "summary": "Could not generate recommendations due to error"
            }