import pandas as pd
import numpy as np
from scipy import stats


class PreprocessorCritique:
    def __init__(self, original_csv, preprocessed_csv, compare_columns):
        self.original_df = pd.read_csv(original_csv)
        self.processed_df = pd.read_csv(preprocessed_csv)
        self.compare_columns = compare_columns

    def compare_distribution(self):
        distribution_result = {}

        for column, col_type in self.compare_columns.items():
            dist_res = "Error"
            reason = "Not evaluated"

            try:
                original = self.original_df[column].dropna()
                processed = self.processed_df[column].dropna()

                if col_type == "numerical continuous":
                    stat, p_value = stats.ks_2samp(original, processed)
                    dist_res = "Same" if p_value > 0.05 else "Different"
                    reason = f"KS test p-value = {p_value:.4f}"

                elif col_type == "numerical discrete":
                    stat, p_value = stats.mannwhitneyu(original, processed, alternative='two-sided')
                    dist_res = "Same" if p_value > 0.05 else "Different"
                    reason = f"Mann–Whitney U test p-value = {p_value:.4f}"

                elif col_type == "categorical nominal":
                    original = original.astype(str)
                    processed = processed.astype(str)

                    orig_counts = np.unique(original, return_counts=True)
                    proc_counts = np.unique(processed, return_counts=True)

                    categories = list(set(orig_counts[0]) | set(proc_counts[0]))
                    orig_freq = [dict(zip(*orig_counts)).get(cat, 0) for cat in categories]
                    proc_freq = [dict(zip(*proc_counts)).get(cat, 0) for cat in categories]

                    stat, p_value = stats.chisquare(f_obs=orig_freq, f_exp=proc_freq)
                    dist_res = "Same" if p_value > 0.05 else "Different"
                    reason = f"Chi-square test p-value = {p_value:.4f}"

                elif col_type == "categorical ordinal":
                    original = original.astype(str)
                    processed = processed.astype(str)

                    unique_values = sorted(set(original) | set(processed))
                    value_to_rank = {val: rank for rank, val in enumerate(unique_values)}

                    original_ranked = np.array([value_to_rank[val] for val in original])
                    processed_ranked = np.array([value_to_rank[val] for val in processed])

                    stat, p_value = stats.mannwhitneyu(original_ranked, processed_ranked, alternative='two-sided')
                    dist_res = "Same" if p_value > 0.05 else "Different"
                    reason = f"Mann–Whitney U test (ordinal) p-value = {p_value:.4f}"

                else:
                    dist_res = "Error"
                    reason = f"Unsupported data type: {col_type}"

            except Exception as e:
                dist_res = "Error"
                reason = str(e)

            distribution_result[column] = {"result": dist_res, "reason": reason}

        return distribution_result
