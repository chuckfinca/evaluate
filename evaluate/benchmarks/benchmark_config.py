
BENCHMARK_CONFIG = {
    'mmlu': {
        'code_url': 'https://github.com/hendrycks/test/archive/refs/heads/master.zip',
        'data_url': 'https://people.eecs.berkeley.edu/~hendrycks/data.tar',
    },
}

def get_benchmark_config(benchmark_name):
    return BENCHMARK_CONFIG.get(benchmark_name)

def get_supported_benchmarks():
    return list(BENCHMARK_CONFIG.keys())