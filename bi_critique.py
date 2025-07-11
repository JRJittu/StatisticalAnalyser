import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import google.generativeai as genai
import os

from kb_statistical import StatisticalKnowledgeBase
from utils import util_functions

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class Bi_critique:
    def __init__(self, knowledge_base: StatisticalKnowledgeBase):
        self.knowledge_base = knowledge_base
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def get_knowledge_for_variables(self, var_type1: str, var_type2: str) -> Dict:
        """Get knowledge base recommendations for bivariate variable types"""
        combined_var_type = f"{var_type1} + {var_type2}"
        doc = self.knowledge_base.search_knowledge("bivariate", combined_var_type)
        if doc == "No relevant statistical test found.":
            raise ValueError(f"No statistical knowledge found for variable types: {combined_var_type}")
        return json.loads(doc)

    def validate_descriptive_statistics(self, data_column1: pd.Series, data_column2: pd.Series,
                                      var_type1: str, var_type2: str, generated_code: str) -> Tuple[bool, str]:
        """
        Validate bivariate descriptive statistics against knowledge base recommendations
        """
        try:
            knowledge = self.get_knowledge_for_variables(var_type1, var_type2)

            # Execute the generated code to get results
            local_vars = {'data_column1': data_column1, 'data_column2': data_column2}
            exec(generated_code, {}, local_vars)

            result = local_vars.get('result')
            if result is None:
                return False, "Generated code did not produce 'result' variable"

            # Convert to serializable format
            serialized_result = util_functions.convert_to_serializable(result)

            # Get knowledge base requirements
            priority_tests = knowledge.get("priority_tests", [])
            descriptive_stats = knowledge.get("descriptive", {}).get("statistics", [])

            validation_prompt = f"""
            You are a statistical validation expert. Validate whether the calculated bivariate statistics match the knowledge base requirements.

            Knowledge Base Requirements:
            Priority Tests: {json.dumps(priority_tests, indent=2)}
            Descriptive Statistics: {json.dumps(descriptive_stats, indent=2)}

            Calculated Results:
            {json.dumps(serialized_result, indent=2)}

            Data Characteristics:
            - Sample size: {len(data_column1)}
            - Variable 1 type: {var_type1}
            - Variable 2 type: {var_type2}
            - Variable 1 nulls: {data_column1.isnull().any()}
            - Variable 2 nulls: {data_column2.isnull().any()}

            Instructions:
            - Check if all required priority tests from knowledge base are calculated for bivariate analysis
            - Check if required descriptive statistics from knowledge base are calculated for bivariate analysis
            - Verify the calculations are appropriate for the bivariate data characteristics
            - Consider the relationship between the two variables
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
            return False, f"Bivariate descriptive validation error: {str(e)}"

    def validate_visualizations(self, data_column1: pd.Series, data_column2: pd.Series,
                               var_type1: str, var_type2: str, desc_results: Dict,
                               selected_visualizations: Dict) -> Tuple[bool, str]:
        """
        Validate bivariate visualizations against knowledge base recommendations
        """
        try:
            knowledge = self.get_knowledge_for_variables(var_type1, var_type2)
            visualization_options = knowledge.get("visualization", {})

            validation_prompt = f"""
            You are a bivariate visualization validation expert. Validate whether the selected visualizations are appropriate for bivariate analysis.

            Knowledge Base Visualization Options:
            {json.dumps(visualization_options, indent=2)}

            Selected Visualizations:
            {json.dumps(selected_visualizations, indent=2)}

            Descriptive Statistics Results:
            {json.dumps(desc_results, indent=2)}

            Data Characteristics:
            - Sample size: {len(data_column1)}
            - Variable 1 range: {data_column1.min()} to {data_column1.max()}
            - Variable 2 range: {data_column2.min()} to {data_column2.max()}
            - Variable 1 type: {var_type1}
            - Variable 2 type: {var_type2}

            Instructions:
            - Check if selected visualizations are available in knowledge base options for bivariate analysis
            - Verify if selections are appropriate for the combination of variable types
            - Consider sample size, data ranges, and relationship patterns
            - Assess if visualizations effectively show the relationship between the two variables
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
            return False, f"Bivariate visualization validation error: {str(e)}"

    def validate_inferential_statistics(self, data_column1: pd.Series, data_column2: pd.Series,
                                      var_type1: str, var_type2: str, desc_results: Dict,
                                      selected_tests: Dict, metadata1: str, metadata2: str) -> Tuple[bool, str]:
        """
        Validate bivariate inferential statistics against knowledge base recommendations
        """
        try:
            knowledge = self.get_knowledge_for_variables(var_type1, var_type2)
            inferential_options = knowledge.get("inferential", {})
            available_tests = inferential_options.get("tests", [])
            selection_criteria = inferential_options.get("selection_criteria", [])

            validation_prompt = f"""
            You are a bivariate inferential statistics validation expert. Validate whether the selected tests are appropriate for bivariate analysis.

            Knowledge Base Inferential Tests:
            Available Tests: {json.dumps(available_tests, indent=2)}
            Selection Criteria: {json.dumps(selection_criteria, indent=2)}

            Selected Tests:
            {json.dumps(selected_tests, indent=2)}

            Descriptive Statistics Results:
            {json.dumps(desc_results, indent=2)}

            Metadata:
            Variable 1: {metadata1}
            Variable 2: {metadata2}

            Data Characteristics:
            - Sample size: {len(data_column1)}
            - Variable 1 type: {var_type1}
            - Variable 2 type: {var_type2}

            Instructions:
            - Check if selected tests are available in the knowledge base for bivariate analysis
            - Verify if test selections meet the selection criteria for the combination of variable types
            - Consider bivariate data characteristics (correlation, independence, etc.) from descriptive results
            - Validate that assumptions for selected bivariate tests are met
            - Check if hypotheses are properly formulated for bivariate relationships
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
            return False, f"Bivariate inferential validation error: {str(e)}"

    def comprehensive_validate(self, data_column1: pd.Series, data_column2: pd.Series,
                             var_type1: str, var_type2: str, generated_code: str,
                             desc_results: Dict, selected_visualizations: Dict,
                             selected_tests: Dict, metadata1: str, metadata2: str,
                             max_retries: int = 3) -> bool:
        """
        Perform comprehensive validation of the entire bivariate analysis
        """
        print("\n" + "="*60)
        print("STARTING BIVARIATE KNOWLEDGE BASE VALIDATION")
        print("="*60)

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n>>> BIVARIATE VALIDATION RETRY ATTEMPT {attempt} <<<")

            all_passed = True

            # Validate descriptive statistics
            print("\n🔍 VALIDATING BIVARIATE DESCRIPTIVE STATISTICS...")
            desc_passed, desc_summary = self.validate_descriptive_statistics(
                data_column1, data_column2, var_type1, var_type2, generated_code
            )

            if desc_passed:
                print("✅ Bivariate descriptive statistics validation PASSED")
            else:
                print("❌ Bivariate descriptive statistics validation FAILED")
                print(f"   {desc_summary}")
                all_passed = False

            # Validate visualizations
            print("\n🔍 VALIDATING BIVARIATE VISUALIZATIONS...")
            viz_passed, viz_summary = self.validate_visualizations(
                data_column1, data_column2, var_type1, var_type2, desc_results, selected_visualizations
            )

            if viz_passed:
                print("✅ Bivariate visualization validation PASSED")
            else:
                print("❌ Bivariate visualization validation FAILED")
                print(f"   {viz_summary}")
                all_passed = False

            # Validate inferential statistics
            print("\n🔍 VALIDATING BIVARIATE INFERENTIAL STATISTICS...")
            inf_passed, inf_summary = self.validate_inferential_statistics(
                data_column1, data_column2, var_type1, var_type2, desc_results,
                selected_tests, metadata1, metadata2
            )

            if inf_passed:
                print("✅ Bivariate inferential statistics validation PASSED")
            else:
                print("❌ Bivariate inferential statistics validation FAILED")
                print(f"   {inf_summary}")
                all_passed = False

            if all_passed:
                print("\n✅ All bivariate validations PASSED")
                return True
            else:
                if attempt < max_retries - 1:
                    print(f"\n❌ Bivariate validation failed. Retrying analysis... (Attempt {attempt + 2}/{max_retries})")
                else:
                    print(f"\n❌ Bivariate analysis validation failed after {max_retries} attempts.")

        return False

    def validate_with_recommendations(self, data_column1: pd.Series, data_column2: pd.Series,
                                    var_type1: str, var_type2: str, analysis_results: Dict) -> Dict[str, Any]:
        """
        Validate bivariate analysis and provide specific recommendations for improvement
        """
        try:
            knowledge = self.get_knowledge_for_variables(var_type1, var_type2)

            recommendation_prompt = f"""
            You are a bivariate statistical analysis advisor. Provide specific recommendations for improving the bivariate analysis.

            Knowledge Base for {var_type1} + {var_type2}:
            {json.dumps(knowledge, indent=2)}

            Current Bivariate Analysis Results:
            {json.dumps(analysis_results, indent=2)}

            Data Characteristics:
            - Sample size: {len(data_column1)}
            - Variable 1 range: {data_column1.min()} to {data_column1.max()}
            - Variable 2 range: {data_column2.min()} to {data_column2.max()}
            - Variable 1 has nulls: {data_column1.isnull().any()}
            - Variable 2 has nulls: {data_column2.isnull().any()}

            Instructions:
            - Compare current bivariate analysis with knowledge base recommendations
            - Identify what's missing or inappropriate for bivariate analysis
            - Focus on relationship analysis between the two variables
            - Provide specific, actionable recommendations for bivariate statistics
            - Return ONLY a JSON with this structure:
            {{
                "overall_assessment": "EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR",
                "strengths": ["list of what was done well in bivariate analysis"],
                "weaknesses": ["list of issues or missing elements in bivariate analysis"],
                "specific_recommendations": [
                    {{
                        "category": "descriptive/visualization/inferential",
                        "issue": "description of the bivariate analysis issue",
                        "recommendation": "specific action to improve bivariate analysis",
                        "priority": "HIGH/MEDIUM/LOW"
                    }}
                ],
                "summary": "overall summary of the bivariate analysis quality"
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
                "summary": "Could not generate bivariate recommendations due to error"
            }