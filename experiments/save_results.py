import json

def save_results(results, path="results.json"):

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved to {path}")
