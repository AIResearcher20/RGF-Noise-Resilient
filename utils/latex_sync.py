def save_main_table(results, path="results/tables/main_results.tex"):

    models = list(results.keys())

    with open(path, "w") as f:

        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Model & Mean & Std \\\\\n")
        f.write("\\hline\n")

        for m in models:
            mean = results[m]["mean"] * 100
            std = results[m]["std"] * 100

            f.write(f"{m} & {mean:.2f} & {std:.2f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
