import os
import pandas as pd
import numpy as np

def extract_correctness_results_from(results_dir):
    """
    Extract 'cors' from CSV files in the results directory.
    """
    all_cors = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(results_dir, filename)
            df = pd.read_csv(file_path)
            if 'correct' in df.columns:
                cors = df['correct'].to_numpy()
                all_cors.append(cors)
    return all_cors

def calculate_scores(all_subject_accs, all_cors):
    """
    Calculate the macro and micro average scores.
    
    :param all_subject_accs: List of average accuracies for each subject
    :param all_cors: List of all individual correctness values
    :return: Tuple of (macro_avg, micro_avg)
    """
    macro_avg = np.mean(all_subject_accs)
    micro_avg = np.mean(all_cors)
    return round(macro_avg * 100, 1), round(micro_avg * 100, 1)