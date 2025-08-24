from config import Config

import pandas as pd
import numpy as np


def read_dataset(config: Config):
    dataset = pd.read_csv(config.dataset_path)
    data = dataset[dataset['id'] >= config.start_question]
    return data[['id', 'question']]


def store_hard_soft_labels_on_step(
    q_id: int,
    hard_labels: str,
    soft_labels_probs: dict[int, list[np.ndarray]],
    soft_labels_indices: list[np.ndarray],
    config: Config
):
    # Check sizes
    n_steps = len(soft_labels_indices)
    for T, probs in soft_labels_probs.items():
        if len(probs) != n_steps:
            raise ValueError(f"For temperature {T}, there should be {n_steps} probabilities, but got {len(probs)}")

    for i in range(len(soft_labels_indices)):
        n_inds = len(soft_labels_indices[i])
        for T, probs in soft_labels_probs.items():
            if len(probs[i]) != n_inds:
                raise ValueError(f"For temperature {T}, there should be {n_inds} at index {i}, but got {len(probs[i])}")

    np.savez(
        config.results_dir / f'q_{q_id}.npz',
        indices=np.array(soft_labels_indices, dtype=object),
        hard_labels=np.array([hard_labels], dtype="U"),  # U -- fixed-length Unicode
        **{f"probs_T={T}": np.array(probs, dtype=object) for T, probs in soft_labels_probs.items()}
    )


if __name__ == "__main__":
    data = np.load("./datasets/results_2025_08_23_19_30/q_1.npz", allow_pickle=True)

    print(data.keys())
    indices = [inds for inds in data["indices"]]  # object to list
    print(len(indices))
    print(type(indices[0]), indices[0].shape)

    hard_labels = str(data["hard_labels"][0])  # object to str
    print(type(hard_labels), hard_labels[:32])

    probs = [ps for ps in data["probs_T=1"]]  # object to list
    print(len(probs))
    print(type(probs[0]), probs[0].shape)
