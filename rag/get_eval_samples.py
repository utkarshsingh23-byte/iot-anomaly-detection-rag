from pathlib import Path

import joblib
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
FEATURE_NAMES_PATH = ROOT_DIR / "cnn_lstm_component" / "config" / "feature_names.txt"
SCALER_PATH = ROOT_DIR / "cnn_lstm_component" / "preprocessing" / "scaler.pkl"

SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42
SEQUENCE_LENGTH = 10
CLASS_LABELS = {
    1: "DDoS",
    2: "Botnet",
}


def get_eval_samples():
    """Return fixed held-out DDoS/Botnet windows with named, original-scale features."""
    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "y_test.npy", mmap_mode="r")
    feature_names = FEATURE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    scaler = joblib.load(SCALER_PATH)

    if x_test.ndim != 3 or x_test.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected X_test windows shaped (samples, {SEQUENCE_LENGTH}, features); "
            f"got {x_test.shape}"
        )
    if y_test.ndim != 1 or len(y_test) != len(x_test):
        raise ValueError("X_test and y_test must contain the same number of windows")
    if x_test.shape[2] != len(feature_names):
        raise ValueError(
            f"X_test has {x_test.shape[2]} features; "
            f"the feature list has {len(feature_names)}"
        )
    if scaler.n_features_in_ != len(feature_names):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features; "
            f"the feature list has {len(feature_names)}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    samples = []

    for class_id, label in CLASS_LABELS.items():
        class_indices = np.flatnonzero(y_test == class_id)
        if len(class_indices) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Expected at least {SAMPLES_PER_CLASS} {label} test windows; "
                f"found {len(class_indices)}"
            )

        selected_indices = rng.choice(
            class_indices,
            size=SAMPLES_PER_CLASS,
            replace=False,
        )
        for sample_number, test_index in enumerate(selected_indices, start=1):
            scaled_window = np.asarray(x_test[int(test_index)], dtype=np.float64)
            if not np.isfinite(scaled_window).all():
                raise ValueError(f"Test window {test_index} contains non-finite values")

            human_readable_window = scaler.inverse_transform(scaled_window)
            samples.append({
                "sample_id": f"{label}_{sample_number}",
                "label": label,
                "true_label": label,
                "class_id": class_id,
                "test_index": int(test_index),
                "features": [
                    {
                        name: float(value)
                        for name, value in zip(feature_names, timestep)
                    }
                    for timestep in human_readable_window
                ],
            })

    return samples


if __name__ == "__main__":
    evaluation_samples = get_eval_samples()
    print(f"Samples: {len(evaluation_samples)}")
    for label in CLASS_LABELS.values():
        print(f"{label}: {sum(sample['label'] == label for sample in evaluation_samples)}")
    first_sample = evaluation_samples[0]
    print(
        f"Example: {first_sample['sample_id']} "
        f"(test window {first_sample['test_index']}), "
        f"{len(first_sample['features'])} timesteps, "
        f"{len(first_sample['features'][0])} named features per timestep"
    )
