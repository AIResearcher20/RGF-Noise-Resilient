import matplotlib.pyplot as plt

def plot_noise(noise_results, save_path="results/figures/noise.png"):

    x = [r[0] for r in noise_results]
    y = [r[1] for r in noise_results]

    plt.figure(figsize=(6,4))

    plt.plot(x, y, marker="o", linewidth=2)

    plt.xlabel("Feature Noise Level")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness under Feature Noise")

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
