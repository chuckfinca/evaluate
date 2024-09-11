import argparse
import json
import os
import sys
from evaluate.models.huggingface_model_loader import HuggingFaceModelLoader
from evaluate.orchestrators.mmlu_benchmark_orchestrator import MMLUEvaluationOrchestrator
from evaluate.benchmarks.benchmark_setup import setup_benchmark
from evaluate.benchmarks.benchmark_config import get_supported_benchmarks
from evaluate.utils.import_utils import get_package_name
from evaluate.logging.logger import logger
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
        logger.log.info("Supported benchmarks:")
        for benchmark in get_supported_benchmarks():
            logger.log.info(f"- {benchmark}")
        sys.exit(0)

    if not args.config:
        # TODO: remove once evaluations work
        local_project_root = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(local_project_root, "dev_config.json")):
            args.config = os.path.join(local_project_root, "dev_config.json")
        else:
            logger.log.error("No configuration file provided and dev_config.json not found.")
            sys.exit(1)

    try:
        config = load_config(args.config)
    except json.JSONDecodeError:
        logger.log.error(f"Error decoding JSON from config file: {args.config}")
        sys.exit(1)
    except FileNotFoundError:
        logger.log.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        validate_config(config)
    except ValueError as e:
        logger.log.error(f"Error in configuration: {str(e)}")
        sys.exit(1)

    logger.set_level(config.get('log_level', 'INFO'))

    logger.log.info(f"Running evaluation '{config['benchmark_name']}' with:")
    logger.log.info(f"Model: {config['model_name']}")
    logger.log.info(f"Number of training examples: {config.get('nshot', 0)}")
    logger.log.info(f"Package name: {get_package_name()}")
    
    try:
        setup_benchmark(config['benchmark_name'])
    except Exception as e:
        logger.log.error(f"Error setting up benchmark: {str(e)}")
        sys.exit(1)

    try:
        model = HuggingFaceModelLoader(config['model_name'])
    except ValueError as e:
        logger.log.error(f"Error loading model: {str(e)}")
        sys.exit(1)

    try:
        evaluator = MMLUEvaluationOrchestrator(model.model, model.tokenizer, config)
        evaluator.evaluate()
    except Exception as e:
        logger.log.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)

    logger.log.info("Evaluation completed successfully.")

if __name__ == "__main__":
    main()