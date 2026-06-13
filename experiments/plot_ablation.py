import matplotlib.pyplot as plt

def plot_ablation(ablation_results,
                  save_path="results/figures/ablation.png"):

    labels = list(ablation_results.keys())
    values = [ablation_results[k] * 100 for k in labels]

    plt.figure(figsize=(6,4))

    plt.bar(labels, values)

    plt.ylabel("Accuracy (%)")
    plt.title("Ablation Study")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=20)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
