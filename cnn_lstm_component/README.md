# Extracted CNN-LSTM IoT Anomaly Detection Component

This folder contains the original CNN-LSTM definition and trained checkpoint from `F:\backup\iot-anomaly-detection`, plus the matching fitted scaler, exact ordered feature list, preprocessing source, and a small inference API. No training was run and no source-project files were edited.

## Architecture and checkpoint

`models/cnn_lstm.py` is copied from the source project. Input shape is `(batch, 10, 38)`. It applies Conv1D 38→64 and 64→128 (kernel size 3, padding 1), BatchNorm1d and ReLU after each convolution, MaxPool1d(kernel 2, stride 1, padding 1), and dropout 0.3. Two batch-first single-layer LSTMs use hidden sizes 128 then 64. The final timestep goes through dropout 0.3 and a 64-unit Linear + ReLU + dropout 0.3 + Linear to 3 logits. Softmax is applied during prediction. The model implementation's default CNN/LSTM/filter configuration is retained unchanged.

The original checkpoint is `models/best_model.pth` (source: `models/saved/best_model.pth`). It contains `model_state`, optimizer state, epoch/validation metrics, `n_features=38`, and `n_classes=3`. Loading uses the original state-dict architecture. `config/model_config.json` records the configuration established from the source code and artifacts.

## Dataset and classes

The source project identifies CICIoT2023 as its dataset. Its preprocessing label map defines three output classes: `0 Normal`, `1 DDoS`, and `2 Botnet`. It maps `BenignTraffic` to Normal, the listed DDoS/DoS flood/fragmentation labels to DDoS, and `Mirai-greeth_flood`, `Mirai-greip_flood`, and `Mirai-udpplain` to Botnet; other labels are filtered out. The source repository README describes ~507,000 sampled rows across 169 CSV files. Raw data is not copied.

## Features and preprocessing

The exact 38 model inputs, in order, are in `config/feature_names.txt` and are also reproduced here:

`Variance, ack_flag_number, Protocol Type, TCP, HTTPS, Duration, syn_count, Min, rst_count, Std, Radius, urg_count, Max, flow_duration, UDP, Covariance, Header_Length, Magnitue, HTTP, ack_count, AVG, Tot size, Tot sum, psh_flag_number, Rate, Srate, DNS, IPv, LLC, fin_count, ARP, Number, Weight, IAT, ICMP, syn_flag_number, rst_flag_number, fin_flag_number`.

`preprocessing/preprocess.py` is the original preprocessing script. It drops identifier columns, keeps numeric features, replaces infinities with NaN, drops columns with >30% missing values, fills remaining NaN values with per-feature medians, removes near-zero variance columns and duplicate rows, selects the top 50 by absolute label correlation (the saved selected feature list has 38), performs stratified 70/15/15 splitting, fits a MinMaxScaler on training rows, and transforms each split. It then creates overlapping stride-1 windows of 10 rows and assigns each window the label of its final row. `preprocessing/scaler.pkl` is the original fitted MinMaxScaler (source: `data/processed/scaler.pkl`).

The source did not serialize the feature medians used to impute missing training values. Therefore `inference/predict.py` orders incoming rows using the saved feature list, applies the saved scaler, and rejects rows with missing, non-numeric, or infinite values rather than guessing imputation values. Input feature names and units/schema should match the CICIoT2023 flow feature definitions. Values are not clipped before scaler transformation, matching `MinMaxScaler.transform` behavior.

## Load and predict

From this directory, install `requirements.txt` in Python 3.11 (matching the source project dependency versions), then:

```python
import pandas as pd
from inference.predict import load_model, predict_rows

model, scaler = load_model()
traffic = pd.read_csv("traffic.csv")  # requires all 38 feature columns
results = predict_rows(traffic, model, scaler)
print(results[0])
```

Or run the CLI:

```bash
python inference/predict.py traffic.csv
```

At least 10 rows are required. Each result corresponds to a stride-1 10-row window and includes the last row index (1-based), predicted class ID/name, predicted-class confidence, and per-class softmax probabilities. CSV label columns are ignored; only the 38 listed feature columns are used.

For parity with the source project's `detection/detector.py`, which expects already-scaled feature rows and does not use the scaler, feed pre-scaled windows directly to the original `CNNLSTM` and use `softmax(logits, dim=1)`. The new API instead accepts raw feature rows and applies the extracted scaler.

## Dependencies

`requirements.txt` includes NumPy, pandas, scikit-learn, joblib, and PyTorch versions from the original project. No dashboard, LLM/RAG, plotting, training, or dataset dependencies are needed to run inference. A CPU installation of PyTorch is sufficient; CUDA is optional.

## Verification and known limits

- Checkpoint, scaler, and ordered feature list were present in the source repository and copied.
- Inference execution and prediction parity were not verified in this environment: the source `venv` points to a Python 3.11 executable that is absent, while the available Python 3.14 has no PyTorch installed. Install the listed supported dependencies to run the supplied inference code.
- The original prediction comparison cannot be run here for the same missing-PyTorch reason.
- The source README and script comments are inconsistent about feature counts (comments mention 50; the actual saved feature file, scaler, preprocessed tensors and checkpoint correspond to 38). This extraction follows the actual artifacts: 38.
- Missing-value medians and raw training data are absent from the extracted component. There is no serialized imputation configuration to reproduce that specific step for incomplete new traffic rows.
