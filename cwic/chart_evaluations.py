
import os
import json
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "eval_results/results"


def get_result_files(base_dir):

    last_files = []

    for root, dirs, files in os.walk(base_dir):
        
        # Check if this is a leaf directory (no subdirectories)
        if not dirs and files:
            
            # Sort files alphabetically and get the last one
            last_file = sorted(files)[-1]
            
            # Get the full path
            full_path = os.path.join(os.path.relpath(root, RESULTS_DIR), last_file)
            last_files.append(full_path)
        
    models = [
        "/".join(d.split("/")[:2]) for d in last_files
    ]

    return list(zip(models, last_files))


def create_table(ax, all_results, metric):
    results = all_results[metric]

    benchmarks = set(list(results.values())[0].keys())
    for v in results.values():
        assert set(v.keys()) == benchmarks, "Not all models have results for the same benchmarks."

    # Get list of models and benchmarks
    models = list(results.keys())
    benchmarks = list(benchmarks)

    models.sort()
    benchmarks.sort()

    pretty_benchmarks = []
    for b in benchmarks:
        if "|" in b:
            pretty_benchmarks.append(b.split("|")[1])
        else:
            pretty_benchmarks.append(b)

    # Create data for the table
    data = []
    for model in models:
        row = []
        for benchmark in benchmarks:
            row.append(round(results[model][benchmark] * 100, 1))
        data.append(row)

    # Convert to numpy array for easier column operations
    data_np = np.array(data)

    # Create table with data
    table = ax.table(
        cellText=[[f"{val:.1f}" for val in row] for row in data],
        rowLabels=models,
        colLabels=pretty_benchmarks,
        loc='center',
        cellLoc='center',
    )

    # Bold the maximum value in each column
    for j in range(len(benchmarks)):
        max_val = max(data_np[:, j])
        for i in range(len(models)):
            if data_np[i, j] == max_val:
                cell = table[(i + 1, j)]  # +1 for the header row
                cell.get_text().set_weight('bold')

    # Make column headers bold
    for j in range(len(benchmarks)):
        cell = table[(0, j)]  # Column headers are in the first row
        cell.get_text().set_weight('bold')
    
    # Make row headers bold
    for i in range(len(models)):
        cell = table[(i + 1, -1)]  # Row headers are in the first column
        cell.get_text().set_weight('bold')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    table.auto_set_column_width(col=list(range(len(benchmarks))))

    ax.set_title(f'Model Performance Comparison ({metric})', fontsize=14, pad=20)


def main():
    
    all_results = {}
    for metric in ["acc_norm", "acc"]:

        results = {}
        for model, file in get_result_files(RESULTS_DIR):

            with open(os.path.join(RESULTS_DIR, file), "r") as f:
                data = json.load(f)
            
            curr_results = {
                benchmark: result[metric] for benchmark, result in data["results"].items()
            }

            results[model] = curr_results
        
        all_results[metric] = results

    num_models = len(all_results["acc"])
    num_benchmarks = len(list(all_results["acc"].values())[0])

    # Create figure and axis
    fig, ax = plt.subplots(2, 1, figsize=(num_benchmarks*1.2 + 2, num_models*0.6 + 2))
    for a in ax:
        a.axis('off')

    create_table(ax[0], all_results, metric="acc")
    create_table(ax[1], all_results, metric="acc_norm")

    plt.tight_layout()
    plt.savefig('model_benchmark_comparison.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()