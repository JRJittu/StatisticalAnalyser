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
from dotenv import load_dotenv

load_dotenv()
UPLOAD_DIR = "uploads"

class CoreAgent:
    def __init__(self):
        self.stat_kb = StatisticalKnowledgeBase(persist_dir='stat_kb_dir')
        self.preprocess_kb = PreprocessorKB(persist_dir='preprocess_kb_dir')

    def analyse_dataset(self, file_path: str, file_name, data_context: str):
        self.data_context = data_context
        self.file_path = file_path
        self.file_name = os.path.splitext(file_name)[0]
        self.dataset = pd.read_csv(file_path)
        self.dataset_pre = None
        self.stat_kb.load_knowledge('uni_bi_kb.json')
        self.preprocess_kb.load_knowledge('preprocess_kb.json')
        self.column_data_type = type_detector.detect_datatypes(self.dataset)
        print("\ntype detector: ", self.column_data_type)
    
        self.data_preprocessing(self.dataset, data_context)
        self.univariate_analysis()
        self.bivariate_analysis()
        print("\n\nANALYSIS DONE. SENDING TO QUERY AGENT\n\n")
        self.combine_result()

        return self.result_output_path, self.selected_data_types, self.selected_pairs

    def data_preprocessing(self, dataset: pd.DataFrame, data_context: str):
        print("\nSTART_PREPROECSSING")
        preprocess_agent = PreprocessorAgent(self.preprocess_kb)

        self.metadata = preprocess_agent.metadata_generator(self.column_data_type, data_context)
        self.selected_data_types = preprocess_agent.feature_remover(self.column_data_type, self.metadata, data_context)
        self.outlier_result = {}
        self.dataset_pre = pd.DataFrame()

        self.selected_data_types = {
            col: dtype for col, dtype in self.selected_data_types.items()
            if dataset[col].isnull().mean() <= 0.3
        }
        print("\nSelected columns: ", self.selected_data_types)
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

        self.processed_file_path = os.path.join(UPLOAD_DIR, f"{self.file_name}_pre.csv")
        self.dataset_pre.to_csv(self.processed_file_path, index=False)

        print("\noutlier_result: \n", self.outlier_result)
        preprocess_critique = PreprocessorCritique(self.file_path, self.processed_file_path, self.selected_data_types)
        self.distribution_result = preprocess_critique.compare_distribution()
        print("\nPreprocess Critique Result: \n", self.distribution_result)
        print("\nEND_PREPROECSSING")


    def univariate_analysis(self):
        print("\nSTART UNIVARIATE\n")
        uni_analyser = UnivariateAnalyzer(self.stat_kb)
        # uni_critique = UniCritique(self.stat_kb)
        self.uni_desc_result = {}
        self.uni_visual_result = {}
        self.uni_inferential_result = {}

        for col, col_type in self.selected_data_types.items():
            desc_result, vis_result, inf_result = uni_analyser.analyze(self.dataset_pre[col], col_type, self.metadata[col], col)
            # desc_result_v, vis_result_v, inf_result_v = uni_critique.validate(self.dataset_pre[col],col_type, self.metadata[col], col, desc_result, vis_result, inf_result)
            
            self.uni_desc_result[col] = desc_result
            self.uni_visual_result[col] = vis_result
            self.uni_inferential_result[col] = inf_result
            
        print("\nUNI DESC RESULT: ")
        for k, v in self.uni_desc_result.items():
            print(k, " : ", v)
        print("\nUNI VISUAL RESULT: ")
        for k, v in self.uni_visual_result.items():
            print(k, " : ", v)
        print("\nUNI INF RESULT: ")
        for k, v in self.uni_inferential_result.items():
            print(k, " : ", v)
            
        print("\nEND UNIVARIATE\n")

    def bivariate_analysis(self):
        print("\nSTART BIVARIATE\n")
        bi_selector = BivariateSelectorAgent(self.selected_data_types)
        bi_analyser = BivariateAnalyzer(self.stat_kb)
        # bi_critique = BiCritique(self.stat_kb)

        self.selected_pairs = bi_selector.select_bivariate_pairs(self.processed_file_path, self.data_context)
        print("\nSelected pairs: ", self.selected_pairs)
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

            # desc_result_v, vis_result_v, inf_result_v = bi_critique.validate(
            #     self.dataset_pre[col1], self.selected_data_types[col1], self.metadata[col1], col1,
            #     self.dataset_pre[col2], self.selected_data_types[col2], self.metadata[col2], col2,
            #     desc_result, vis_result, inf_result
            # )

            combine = col1 + "-" + col2
            self.bi_desc_result[combine] = desc_result
            self.bi_visual_result[combine] = vis_result
            self.bi_inferential_result[combine] = inf_result
        
        print("\nBI DESC RESULT: ")
        for k, v in self.bi_desc_result.items():
            print(k, " : ", v)
            
        print("\nBI VISUAL RESULT: ")
        for k, v in self.bi_visual_result.items():
            print(k, " : ", v)
        print("\nBI INF RESULT: ")
        for k, v in self.bi_inferential_result.items():
            print(k, " : ", v)

        print("\nEND BIVARIATE\n")

    def combine_result(self):
        combined_dict = {
            "preprocessing": {
                "outlier_result": self.outlier_result,
                "distribution_result": self.distribution_result
            },
            "univariate": {
                "descriptive": self.uni_desc_result,
                "visual": self.uni_visual_result,
                "inferential": self.uni_inferential_result
            },
            "bivariate": {
                "descriptive": self.bi_desc_result,
                "visual": self.bi_visual_result,
                "inferential": self.bi_inferential_result
            }
        }

        self.result_output_path = os.path.join(UPLOAD_DIR, f"{self.file_name}_result.json")
        with open(self.result_output_path, "w", encoding="utf-8") as f:
            json.dump(combined_dict, f, indent=2)
