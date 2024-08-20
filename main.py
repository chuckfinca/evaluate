"""
This project uses the Measuring Massive Multitask Language Understanding (MMLU) benchmark.
MMLU Citation: Hendrycks et al. (2021). Measuring Massive Multitask Language Understanding. ICLR 2021.
Ethics Citation: Hendrycks et al. (2021). Aligning AI With Shared Human Values. ICLR 2021.
"""
import argparse
import os
import sys
from models.huggingface_model import HuggingFaceModel
from evaluation.evaluator_mmlu import MMLUEvaluator
from config import add_to_sys_path, get_evaluation_project_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a specified benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = argparse.ArgumentParser(description="Evaluate a model on a specified benchmark")
    parser.add_argument("--evaluation", type=str, help="Name of the evaluation (e.g., 'mmlu', 'hellaswag')")
    parser.add_argument("--model", type=str, help="Name or path of the model to evaluate")
    parser.add_argument("--data_dir", type=str, 
                        help="Path to the data directory (default: benchmarks/<evaluation>/data)")
    parser.add_argument("--save_dir", type=str, 
                        help="Directory to save evaluation results (default: ./results/<evaluation>)")
    parser.add_argument("--ntrain", type=int, default=5, help="Number of examples to use for few-shot learning")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for generated sequences")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")

    return parser.parse_args()

def main(args):
    # Handle the case where no evaluation is provided
    if args.evaluation is None:
        print("No evaluation specified.")

    # Set default paths based on the evaluation name if not provided
    if args.data_dir is None:
        args.model = "google/gemma-2b-it"
    if args.data_dir is None:
        args.data_dir = os.path.join(os.getcwd(), "benchmarks", "data", args.evaluation)
    if args.save_dir is None:
        args.save_dir = os.path.join(os.getcwd(), "results", args.evaluation)

    # Add the specific benchmark/evaluation code's folder to sys.path
    path = get_evaluation_project_path(args.evaluation)
    add_to_sys_path(path)

    print(f"Running evaluation '{args.evaluation}' with:")
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Number of training examples: {args.ntrain}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    
    model = HuggingFaceModel(args.model)
    evaluator = MMLUEvaluator(model, args)
    evaluator.evaluate()

if __name__ == "__main__":
    args = parse_args()
    main(args)