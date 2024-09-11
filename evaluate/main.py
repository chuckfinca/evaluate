import argparse
import json
import logging
import os
import sys
from evaluate.models.huggingface_model_loader import HuggingFaceModelLoader
from evaluate.orchestrators.mmlu_benchmark_orchestrator import MMLUEvaluationOrchestrator
from evaluate.benchmarks.benchmark_setup import setup_benchmark
from evaluate.benchmarks.benchmark_config import get_supported_benchmarks
from evaluate.utils.import_utils import get_package_name
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a specified benchmark using a configuration file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", type=str, nargs='?', help="Path to JSON configuration file")
    parser.add_argument("--list_benchmarks", action="store_true", help="List all supported benchmarks and exit")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_config(config):
    required_params = ['benchmark_name', 'model_name']
    missing_params = [param for param in required_params if param not in config]
    
    if missing_params:
        raise ValueError(f"Missing required parameters in config: {', '.join(missing_params)}")
    
    if config['benchmark_name'] not in get_supported_benchmarks():
        raise ValueError(f"Unsupported benchmark: {config['benchmark_name']}")

def main():
    args = parse_args()

    if args.list_benchmarks:
        print("Supported benchmarks:")
        for benchmark in get_supported_benchmarks():
            print(f"- {benchmark}")
        sys.exit(0)

    if not args.config:
        # TODO: remove once evaluations work
        local_project_root = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(local_project_root, "dev_config.json")):
            args.config = os.path.join(local_project_root, "dev_config.json")

    config = load_config(args.config)

    try:
        validate_config(config)
    except ValueError as e:
        print(f"Error in configuration: {str(e)}")
        sys.exit(1)

    setup_logging(config.get('log_level', 'INFO'))

    logging.info(f"Running evaluation '{config['benchmark_name']}' with:")
    logging.info(f"Model: {config['model_name']}")
    logging.info(f"Number of training examples: {config.get('nshot', 0)}")
    logging.info(f"Package name: {get_package_name()}")
    
    setup_benchmark(config['benchmark_name'])

    try:
        model = HuggingFaceModelLoader(config['model_name'])
    except ValueError as e:
        logging.error(f"Error loading model: {str(e)}")
        sys.exit(1)

    evaluator = MMLUEvaluationOrchestrator(model.model, model.tokenizer, config)
    evaluator.evaluate()

if __name__ == "__main__":
    main()