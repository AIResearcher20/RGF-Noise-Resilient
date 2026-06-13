import matplotlib.pyplot as plt

def plot_accuracy(results, save_path="results/figures/accuracy.png"):

    models = list(results.keys())
    means = [results[m]["mean"] * 100 for m in models]
    stds = [results[m]["std"] * 100 for m in models]

    plt.figure(figsize=(6,4))

    plt.bar(models, means, yerr=stds, capsize=5)

    plt.ylabel("Accuracy (%)")
    plt.title("Node Classification Performance")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
