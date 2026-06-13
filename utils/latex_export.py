def results_to_latex(results):

    print("\n\\begin{tabular}{lcc}")
    print("\\hline")
    print("Model & Mean & Std \\\\")
    print("\\hline")

    for model, res in results.items():
        print(f"{model} & {res['mean']*100:.2f} & {res['std']*100:.2f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
