from utils.noise import add_feature_noise
from train.train_eval import train_eval

def noise_experiment(model_class, X, A, y, train_mask, val_mask, test_mask):
    levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for n in levels:
        Xn = add_feature_noise(X, n)

        model = model_class(X.shape[1], 64, int(y.max()+1))

        acc, _ = train_eval(model, Xn, A, y, train_mask, val_mask, test_mask)

        results.append(acc)

        print(f"Noise {n}: {acc:.4f}")

    return results
