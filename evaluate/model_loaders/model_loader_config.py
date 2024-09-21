MODEL_LOADER_CONFIG = {
    'meta-llama/Meta-Llama-3.1-8B-Instruct': {
        'bos_id': 128000, # '<|begin_of_text|>',
        'eos_id': 128001, #'<|end_of_text|>',
        'pad_id': -1, # Llama 3.1 doesn't have one
        'stop_tokens': set([128001, 128009]) #['<|end_of_text|>', '<|eot_id|>'],
    },
}

def get_model_loader_config(benchmark_name):
    return MODEL_LOADER_CONFIG.get(benchmark_name)