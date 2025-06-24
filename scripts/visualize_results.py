import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RESULTS_DIR = "../results"  

def load_evaluation_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                model = os.path.splitext(filename)[0]
                for task_name, task_data in data["task_metrics"].items():
                    metrics = task_data["metrics"]
                    all_data.append({
                        "model": model,
                        "Task": task_name,
                        "f1_supported": metrics["supported"]["f1"],
                        "f1_hallucinated": metrics["hallucinated"]["f1"],
                        "auroc": metrics["auroc"],
                        "runtime [s]": task_data["runtime_seconds"]
                    })
    return pd.DataFrame(all_data)

def plot_and_save(df, metric, title, filename):
    plt.figure(figsize=(12, 8))  # Increased height from 6 to 8
    sns.barplot(data=df, x="Task", y=metric, hue="model")
    plt.title(title)
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space for legend on right

    # Move legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    output_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def main():
    df = load_evaluation_files(RESULTS_DIR)

    print("Loaded metrics summary:")
    print(df.head())

    plot_and_save(df, "f1_supported", "F1 Score (Supported) by Task and Model", "f1_supported.png")
    plot_and_save(df, "f1_hallucinated", "F1 Score (Hallucinated) by Task and Model", "f1_hallucinated.png")
    plot_and_save(df, "auroc", "AUROC by Task and Model", "auroc.png")
    plot_and_save(df, "runtime [s]", "Runtime (seconds) by Task and Model", "runtime.png")

if __name__ == "__main__":
    main()
