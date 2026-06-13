import matplotlib.pyplot as plt

def plot_gate(gate_values,
              save_path="results/figures/gate.png"):

    plt.figure(figsize=(6,4))

    plt.hist(gate_values, bins=30, alpha=0.8)

    plt.xlabel("Gate Value (g)")
    plt.ylabel("Number of Nodes")
    plt.title("RGF Gate Distribution")

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
