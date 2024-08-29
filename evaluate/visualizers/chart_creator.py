import matplotlib.pyplot as plt
import os
import time

from processors.result_processor import path_to_results

def create_mmlu_comparison_chart(realized_score, reported_score, args):
    """
    Create a bar chart comparing current and previous MMLU scores.

    :param realized_score: float, the current evaluation score
    :param reported_score: float, the previous evaluation score
    :param model_name: str, the name of the model being evaluated
    :param output_dir: str, the directory to save the chart
    """
    # Generate output filename
    timestamp = int(time.time())
    output_filename = f"reported_realized_{timestamp}.png"
    output_directory = path_to_results(args, False)

    # path = os.path.join(args.project_root, "benchmarks", args.benchmark_name, "results", args.model_name)
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, output_filename)


    # Data
    evaluation = ['Reported', 'Realized']
    scores = [reported_score, realized_score]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(evaluation, scores, color=['#D3D3D3', '#4285F4'])

    # Customize the chart
    ax.set_ylabel('MMLU Score', fontsize=18)
    ax.set_title(f'MMLU Test Results: {args.model_name}', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 100)  # Set y-axis from 0 to 100 for percentage scale

    # x-axis labels
    ax.set_xticklabels(evaluation, fontsize=18)

    # y-axis label
    ax.tick_params(axis='y', labelsize=18)

    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=18)

    # Add a light grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Improve overall layout
    plt.tight_layout()


    # Save the chart
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Chart saved as {output_path}")