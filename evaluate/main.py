"""
This project uses the Measuring Massive Multitask Language Understanding (MMLU) benchmark.
MMLU Citation: Hendrycks et al. (2021). Measuring Massive Multitask Language Understanding. ICLR 2021.
Ethics Citation: Hendrycks et al. (2021). Aligning AI With Shared Human Values. ICLR 2021.
"""
import argparse
import json
import os
import sys
from models.huggingface_model import HuggingFaceModel
from orchestrators.mmlu_benchmark_orchestrator import MMLUBenchmarkOrchestrator
from config import add_to_sys_path, get_evaluation_project_path
from benchmarks.benchmark_setup import setup_benchmark

from dotenv import load_dotenv
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a specified benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--benchmark_name", type=str, help="Name of the benchmark (e.g., 'mmlu', 'hellaswag')")
    parser.add_argument("--model_name", type=str, help="Name or path of the model to evaluate")
    parser.add_argument("--project_root", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Path to the project folder")
    parser.add_argument("--ntrain", type=int, default=5, help="Number of examples to use for few-shot learning")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for generated sequences")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def check_required_args(args):
    required_args = ['benchmark_name', 'model_name', 'project_root']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")
    
    # Check if main.py exists in the project root
    main_py_path = os.path.join(args.project_root, 'main.py')
    if not os.path.isfile(main_py_path):
        raise ValueError(f"Invalid project root: {args.project_root}. Please provide the path to the project root where main.py can be found.")



def main(args):

    project_root = args.project_root

    # TODO: remove once evaluations work
    if os.path.exists(os.path.join(project_root, "test_config.py")):
        args.config = os.path.join(project_root, "test_config.py")

        # Set args from config if supplied
        if args.config:
            config = load_config(args.config)
            for key, value in config.items():
                setattr(args, key, value)
    
    # Check for required arguments after potentially loading from config
    check_required_args(args)

    # Set up the benchmark if it's not already present
    setup_benchmark(args.benchmark_name, project_root)

    # Add the specific benchmark/evaluation code's folder to sys.path
    path = get_evaluation_project_path(args.benchmark_name)
    add_to_sys_path(path)

    print(f"Running evaluation '{args.benchmark_name}' with:")
    print(f"Model: {args.model_name}")
    print(f"Number of training examples: {args.ntrain}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    
    model = HuggingFaceModel(args.model_name, project_root, args.device)
    evaluator = MMLUBenchmarkOrchestrator(model.model, model.tokenizer, args, project_root)
    evaluator.evaluate()

if __name__ == "__main__":
    args = parse_args()
    main(args)