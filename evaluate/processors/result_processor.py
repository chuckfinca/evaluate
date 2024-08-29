import os
import pandas as pd
import numpy as np

def path_to_results(args, raw):
    path = os.path.join(args.project_root, "benchmarks", "benchmarks", args.benchmark_name, "results", args.model_name)
    if raw:
        path = os.path.join(path, 'raw')

    os.makedirs(path, exist_ok=True)
    return path

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

def calculate_score(all_cors):
    """
    Calculate the average score from all cors.
    """
    return round(np.mean(np.concatenate(all_cors)) * 100, 1)