import argparse
import json
import os
import sys
from evaluate.models.huggingface_model import HuggingFaceModel
from evaluate.orchestrators.mmlu_benchmark_orchestrator import MMLUBenchmarkOrchestrator
from evaluate.benchmarks.benchmark_setup import setup_benchmark
from evaluate.benchmarks.benchmark_config import get_supported_benchmarks

from dotenv import load_dotenv

from evaluate.utils.module_utils import get_package_name
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a specified benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--benchmark_name", type=str, help="Name of the benchmark (e.g., 'mmlu'). Use --list_benchmarks to see supported benchmarks.")
    parser.add_argument("--list_benchmarks", action="store_true", help="List all supported benchmarks and exit")
    parser.add_argument("--model_name", type=str, help="Name or path of the model to evaluate")
    parser.add_argument("--nshot", type=int, default=0, help="Number (n) of examples to use for n-shot learning")
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
    local_project_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(local_project_root, "test_config.py")):
        args.config = os.path.join(local_project_root, "test_config.py")

        # Set args from config if supplied
        if args.config:
            config = load_config(args.config)
            for key, value in config.items():
                setattr(args, key, value)
    
    # Check for required arguments after potentially loading from config
    check_required_args(args)

    # Set up the benchmark if it's not already present
    setup_benchmark(args.benchmark_name)

    print(f"Running evaluation '{args.benchmark_name}' with:")
    print(f"Model: {args.model_name}")
    print(f"Number of training examples: {args.nshot}")
    print(f"Package name: {get_package_name()}")
    
    model = HuggingFaceModel(args.model_name)
    evaluator = MMLUBenchmarkOrchestrator(model.model, model.tokenizer, args.benchmark_name, args.model_name, args.nshot)
    evaluator.evaluate()

if __name__ == "__main__":
    args = parse_args()
    main(args)