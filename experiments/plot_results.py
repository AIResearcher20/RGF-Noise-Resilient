import matplotlib.pyplot as plt

def plot_noise(results):

    x = [r[0] for r in results]
    y = [r[1] for r in results]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title("Noise Robustness - RGF")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
