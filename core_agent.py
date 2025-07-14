import pandas as pd
import google.generativeai as genai
import os
import json

from kb_preprocess import PreprocessorKB
from kb_statistical import StatisticalKnowledgeBase
from preprocess_agent import PreprocessorAgent
from preprocess_critique import PreprocessorCritique
from uni_agent import UnivariateAnalyzer
from uni_critique import UniCritique
from bi_selector import BivariateSelectorAgent
from bi_agent import BivariateAnalyzer
from bi_critique import BiCritique
import type_detector


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = "uploads"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

class CoreAgent:
    def __init__(self):
        self.stat_kb = StatisticalKnowledgeBase(persist_dir='stat_kb_dir')
        self.preprocess_kb = PreprocessorKB(persist_dir='preprocess_kb_dir')

    def analyse_dataset(self, file_path: str, file_name, data_context: str):
        self.file_path = file_path
        self.file_name = file_name
        self.dataset = pd.read_csv(file_path)
        self.dataset_pre = None
        self.column_data_type = type_detector.detect_datatypes(self.dataset, model)
        print("\ntype detector: ", self.column_data_type)
    
        self.data_preprocessing(self.dataset, self.file_name, data_context)
        self.univariate_analysis()
        self.bivariate_analysis()
        print("\n\nANALYSIS DONE. SENDING TO QUERY AGENT\n\n")
        self.combine_result()

        return self.result_output_path

    def data_preprocessing(self, dataset: pd.DataFrame, file_name: str, data_context: str):
        preprocess_agent = PreprocessorAgent(self.preprocess_kb, GOOGLE_API_KEY)

        self.metadata = preprocess_agent.metadata_generator(self.column_data_type, data_context)
        self.selected_data_types = preprocess_agent.feature_remover(self.column_data_type, self.metadata, data_context)
        self.outlier_result = {}
        self.dataset_pre = pd.DataFrame()

        self.selected_data_types = {
            col: dtype for col, dtype in self.selected_data_types.items()
            if dataset[col].isnull().mean() <= 0.3
        }

        for column, col_type in self.selected_data_types.items():
            preprocess_agent.fetch_knowledge(col_type)
            out_result = preprocess_agent.outlier_detector(data_column=dataset[column], data_type=col_type, metadata=self.metadata[column])
            self.outlier_result[column] = out_result

            if dataset[column].isnull().any():
                miss_val_result = preprocess_agent.missing_value_imputer(data_column=dataset[column], data_type=col_type, metadata=self.metadata[column])

                if "imputed_data" in miss_val_result:
                    self.dataset_pre[column] = miss_val_result["imputed_data"]
                else:
                    self.dataset_pre[column] = dataset[column]
            else:
                self.dataset_pre[column] = dataset[column]
                print(f"No missing values in column: {column}")

        self.processed_file_path = os.path.join(UPLOAD_DIR, f"{file_name}_pre.csv")
        self.dataset_pre.to_csv(self.processed_file_path, index=False)

        print("\noutlier_result: \n", self.outlier_result)
        preprocess_critique = PreprocessorCritique(self.file_path, self.processed_file_path, self.selected_data_types)
        self.distribution_result = preprocess_critique.compare_distribution()
        print("\nPreprocess Critique Result: \n", self.distribution_result)


    def univariate_analysis(self):
        uni_analyser = UnivariateAnalyzer(self.stat_kb, GOOGLE_API_KEY)
        uni_critique = UniCritique(self.stat_kb, GOOGLE_API_KEY)
        self.uni_desc_result = {}
        self.uni_visual_result = {}
        self.uni_inferential_result = {}

        for col, col_type in self.selected_data_types.items():
            desc_result, vis_result, inf_result = uni_analyser.analyze(self.dataset_pre[col], col_type, self.metadata[col], col)

            desc_result_v, vis_result_v, inf_result_v = uni_critique.validate(self.dataset_pre[col],col_type, self.metadata[col], col, desc_result, vis_result, inf_result)

            self.uni_desc_result[col] = desc_result_v
            self.uni_visual_result[col] = vis_result_v
            self.uni_inferential_result[col] = inf_result_v


    def bivariate_analysis(self):
        bi_selector = BivariateSelectorAgent(self.selected_data_types, BiCritique)
        bi_analyser = BivariateAnalyzer(self.stat_kb, GOOGLE_API_KEY)
        bi_critique = BiCritique(self.stat_kb, GOOGLE_API_KEY)

        self.selected_pairs = bi_selector.select_bivariate_pairs(self.processed_file_path)

        self.bi_desc_result = {}
        self.bi_visual_result = {}
        self.bi_inferential_result = {}

        for temp in self.selected_pairs:
            col1 = temp['pair'][0]
            col2 = temp['pair'][1]

            desc_result, vis_result, inf_result = bi_analyser.analyze(
                self.dataset_pre[col1], self.selected_data_types[col1], self.metadata[col1], col1,
                self.dataset_pre[col2], self.selected_data_types[col2], self.metadata[col2], col2,
            )

            desc_result_v, vis_result_v, inf_result_v = bi_critique.validate(
                self.dataset_pre[col1], self.selected_data_types[col1], self.metadata[col1], col1,
                self.dataset_pre[col2], self.selected_data_types[col2], self.metadata[col2], col2,
                desc_result, vis_result, inf_result
            )

            combine = col1 + "-" + col2
            self.bi_desc_result[combine] = desc_result_v
            self.bi_visual_result[combine] = vis_result_v
            self.bi_inferential_result[combine] = inf_result_v

    def combine_result(self):
        self.combined = ""

        self.combined += "### Preprocessing Outlier Results:\n"
        self.combined += json.dumps(self.outlier_result, indent=2)
        self.combined += "\n\n### Preprocessing Distribution Comparison:\n"
        self.combined += json.dumps(self.distribution_result, indent=2)

        self.combined += "\n\n### Univariate Descriptive Results:\n"
        self.combined += json.dumps(self.uni_desc_result, indent=2)
        self.combined += "\n\n### Univariate Visual Results:\n"
        self.combined += json.dumps(self.uni_visual_result, indent=2)
        self.combined += "\n\n### Univariate Inferential Results:\n"
        self.combined += json.dumps(self.uni_inferential_result, indent=2)

        self.combined += "\n\n### Bivariate Descriptive Results:\n"
        self.combined += json.dumps(self.bi_desc_result, indent=2)
        self.combined += "\n\n### Bivariate Visual Results:\n"
        self.combined += json.dumps(self.bi_visual_result, indent=2)
        self.combined += "\n\n### Bivariate Inferential Results:\n"
        self.combined += json.dumps(self.bi_inferential_result, indent=2)

        self.result_output_path = os.path.join(UPLOAD_DIR, f"{self.file_name}_result.txt")
        with open(self.result_output_path, "w", encoding="utf-8") as f:
            f.write(self.combined)