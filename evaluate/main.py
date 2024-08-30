"""
This project uses the Measuring Massive Multitask Language Understanding (MMLU) benchmark.
MMLU Citation: Hendrycks et al. (2021). Measuring Massive Multitask Language Understanding. ICLR 2021.
Ethics Citation: Hendrycks et al. (2021). Aligning AI With Shared Human Values. ICLR 2021.
"""
import argparse
import json
import os
import sys
from .config import PROJECT_ROOT
from models.huggingface_model import HuggingFaceModel
from orchestrators.mmlu_benchmark_orchestrator import MMLUBenchmarkOrchestrator
from benchmarks.benchmark_setup import setup_benchmark
from benchmarks.benchmark_config import get_supported_benchmarks
from visualizers.chart_creator import create_mmlu_comparison_chart

from dotenv import load_dotenv
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a specified benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--benchmark_name", type=str, help="Name of the benchmark (e.g., 'mmlu'). Use --list_benchmarks to see supported benchmarks.")
    parser.add_argument("--list_benchmarks", action="store_true", help="List all supported benchmarks and exit")
    parser.add_argument("--model_name", type=str, help="Name or path of the model to evaluate")
    parser.add_argument("--reported_score", type=str, help="The official reported score for the benchmark, if available")
    parser.add_argument("--project_root", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Path to the project folder")
    parser.add_argument("--ntrain", type=int, default=5, help="Number of examples to use for few-shot learning")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for generated sequences")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")

    args = parser.parse_args()

    if args.list_benchmarks:
        print("Supported benchmarks:")
        for benchmark in get_supported_benchmarks():
            print(f"- {benchmark}")
        sys.exit(0)

    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def check_required_args(args):
    required_args = ['benchmark_name', 'model_name']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]

    if missing_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")


def main(args):

    # TODO: remove once evaluations work
    if os.path.exists(os.path.join(PROJECT_ROOT, "test_config.py")):
        args.config = os.path.join(PROJECT_ROOT, "test_config.py")

        # Set args from config if supplied
        if args.config:
            config = load_config(args.config)
            for key, value in config.items():
                setattr(args, key, value)
    
    # Check for required arguments after potentially loading from config
    check_required_args(args)

    # Set up the benchmark if it's not already present
    setup_benchmark(args.benchmark_name, PROJECT_ROOT)

    print(f"Running evaluation '{args.benchmark_name}' with:")
    print(f"Model: {args.model_name}")
    print(f"Reported Score: {args.reported_score}")
    print(f"Number of training examples: {args.ntrain}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    
    model = HuggingFaceModel(args)
    realized_acc = evaluator.evaluate()

    if args.reported_score is not None:
        create_mmlu_comparison_chart(realized_acc, args.reported_score, args)
    evaluator = MMLUBenchmarkOrchestrator(model.model, model.tokenizer, args, PROJECT_ROOT)

if __name__ == "__main__":
    args = parse_args()
    main(args)