import torch

def add_feature_noise(X, noise_level):
    return X + noise_level * torch.randn_like(X)


def add_structural_noise(edge_index, noise_level):
    num_edges = edge_index.shape[1]
    mask = torch.rand(num_edges) > noise_level
    return edge_index[:, mask]


def noise_experiment(model_class, X, A, edge_index, y,
                     train_mask, val_mask, test_mask,
                     noise_levels):

    results = []

    for noise in noise_levels:

        X_noisy = add_feature_noise(X, noise)
        edge_noisy = add_structural_noise(edge_index, noise)

        A_noisy = A  # ساده نگه می‌داریم (فعلاً)

        print(f"Running noise level {noise}")

        results.append(None)  # فعلاً placeholder

    return results
