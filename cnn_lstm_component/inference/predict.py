from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))
from cnn_lstm import CNNLSTM

LABELS = ["Normal", "DDoS", "Botnet"]
SEQUENCE_LENGTH = 10
FEATURES = (ROOT / "config" / "feature_names.txt").read_text(encoding="utf-8").splitlines()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the extracted original checkpoint and its matching scaler."""
    checkpoint = torch.load(ROOT / "models" / "best_model.pth", map_location=DEVICE, weights_only=False)
    model = CNNLSTM(
        n_features=checkpoint.get("n_features", 38),
        n_classes=checkpoint.get("n_classes", 3),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = joblib.load(ROOT / "preprocessing" / "scaler.pkl")
    if len(FEATURES) != model.conv_block[0].in_channels:
        raise ValueError(f"Feature list has {len(FEATURES)} entries; model expects {model.conv_block[0].in_channels}")
    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError(f"Scaler expects {scaler.n_features_in_} columns; feature list has {len(FEATURES)}")
    return model, scaler


def preprocess_rows(frame: pd.DataFrame, scaler):
    """Order exact training features, require complete finite numeric rows, and apply saved MinMaxScaler."""
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError("Input is missing required training features: " + ", ".join(missing))
    values = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Input contains missing, non-numeric, or infinite feature values; training medians were not saved")
    return scaler.transform(values).astype(np.float32)


def make_sequences(scaled_rows: np.ndarray):
    """Overlapping windows of 10 rows; one output for each window ending at its last row."""
    if scaled_rows.ndim != 2 or scaled_rows.shape[1] != len(FEATURES):
        raise ValueError(f"Expected rows with {len(FEATURES)} ordered features")
    if len(scaled_rows) < SEQUENCE_LENGTH:
        raise ValueError(f"Need at least {SEQUENCE_LENGTH} rows; got {len(scaled_rows)}")
    return np.stack([scaled_rows[i:i + SEQUENCE_LENGTH] for i in range(len(scaled_rows) - SEQUENCE_LENGTH + 1)])


@torch.inference_mode()
def predict_rows(frame: pd.DataFrame, model=None, scaler=None):
    """Classify CSV/DataFrame traffic rows. Returns predictions/probabilities per complete window."""
    if model is None or scaler is None:
        model, scaler = load_model()
    sequences = make_sequences(preprocess_rows(frame, scaler))
    logits = model(torch.as_tensor(sequences, dtype=torch.float32, device=DEVICE))
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    class_ids = probabilities.argmax(axis=1)
    return [
        {
            "window_end_row": i + SEQUENCE_LENGTH,
            "class_id": int(class_id),
            "label": LABELS[int(class_id)],
            "confidence": float(probabilities[i, class_id]),
            "probabilities": {LABELS[j]: float(probabilities[i, j]) for j in range(len(LABELS))},
        }
        for i, class_id in enumerate(class_ids)
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CNN-LSTM IoT network anomaly inference")
    parser.add_argument("csv", help="CSV with all 38 named model input features")
    args = parser.parse_args()
    model, scaler = load_model()
    for result in predict_rows(pd.read_csv(args.csv), model, scaler):
        print(result)
