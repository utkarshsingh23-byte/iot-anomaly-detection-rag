import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPONENT_DIR = ROOT_DIR / "cnn_lstm_component"
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "rag"

MODEL_NAMES = ("llama3.2:3b", "gemma2:2b", "phi3:mini")
LABELS = ("Normal", "DDoS", "Botnet")
BLIND_RANDOM_SEED = 2026
OLLAMA_TIMEOUT = (10, 600)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.cnn_lstm import CNNLSTM
from rag.generate import (
    OLLAMA_GENERATE_URL,
    finalize_explanation,
    build_llm_only_prompt,
    build_rag_prompt,
)
from rag.get_eval_samples import get_eval_samples
from rag.retrieve import retrieve


RESULT_FIELDS = [
    "sample_id",
    "true_label",
    "detected_label",
    "confidence",
    "retrieved_sources",
    "llm_only_text",
    "rag_text",
    "llm_only_seconds",
    "rag_seconds",
    "model",
]
BLIND_FIELDS = [
    "blind_id",
    "true_label",
    "detected_label",
    "confidence",
    "explanation_text",
]
ANSWER_KEY_FIELDS = ["blind_id", "sample_id", "method", "model"]


def load_detector():
    checkpoint = torch.load(
        COMPONENT_DIR / "models" / "best_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    model = CNNLSTM(
        n_features=checkpoint.get("n_features", 38),
        n_classes=checkpoint.get("n_classes", 3),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def detect_sample(model, x_test, y_test, sample):
    test_index = sample["test_index"]
    if int(y_test[test_index]) != sample["class_id"]:
        raise ValueError(f"Sample label does not match test data at index {test_index}")

    window = np.array(x_test[test_index], dtype=np.float32, copy=True)
    if not np.isfinite(window).all():
        raise ValueError(f"Test window {test_index} contains non-finite values")

    with torch.inference_mode():
        logits = model(torch.from_numpy(window).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0].numpy()

    class_id = int(probabilities.argmax())
    return LABELS[class_id], float(probabilities[class_id])


def _post_generate(session, payload):
    response = session.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("error"):
        raise RuntimeError(f"Ollama generation failed: {result['error']}")
    return result


def load_llm_model(session, model_name):
    _post_generate(session, {
        "model": model_name,
        "prompt": "",
        "stream": False,
        "keep_alive": -1,
    })


def unload_llm_model(session, model_name):
    _post_generate(session, {
        "model": model_name,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    })


def generate_full_text(session, model_name, prompt):
    result = _post_generate(session, {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
    })
    text = result.get("response")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ollama response did not contain non-empty full text")
    return text


def get_retrieval_query(attack_type, features):
    feature_text = json.dumps(features, indent=2, ensure_ascii=False, default=str)
    return f"IoT network {attack_type} anomaly; observed features: {feature_text}"


def detect_samples(samples, detector, x_test, y_test):
    detections = []
    for sample in samples:
        detected_label, confidence = detect_sample(detector, x_test, y_test, sample)
        detections.append((sample, detected_label, confidence))
    return detections


def evaluate_model(session, model_name, detected_samples):
    rows = []
    load_llm_model(session, model_name)
    try:
        for sample_number, (sample, detected_label, confidence) in enumerate(
            detected_samples,
            start=1,
        ):
            features = sample["features"]

            llm_start = time.perf_counter()
            llm_prompt = build_llm_only_prompt(
                detected_label, features, model=model_name
            )
            llm_only_text = generate_full_text(session, model_name, llm_prompt)
            llm_only_text = finalize_explanation(
                llm_only_text,
                detected_label,
                features,
                model=model_name,
            )
            llm_only_seconds = time.perf_counter() - llm_start

            rag_start = time.perf_counter()
            if detected_label == "Normal":
                chunks = []
                rag_prompt = build_llm_only_prompt(
                    detected_label, features, model=model_name
                )
            else:
                chunks = retrieve(
                    get_retrieval_query(detected_label, features),
                    k=3,
                    attack_type=detected_label,
                )
                rag_prompt = build_rag_prompt(
                    detected_label, features, chunks, model=model_name
                )
            rag_text = generate_full_text(session, model_name, rag_prompt)
            rag_text = finalize_explanation(
                rag_text,
                detected_label,
                features,
                model=model_name,
                retrieved_chunks=chunks,
            )
            rag_seconds = time.perf_counter() - rag_start

            rows.append({
                "sample_id": sample["sample_id"],
                "true_label": sample["label"],
                "detected_label": detected_label,
                "confidence": confidence,
                "retrieved_sources": json.dumps([
                    {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "score": chunk["score"],
                    }
                    for chunk in chunks
                ], ensure_ascii=False),
                "llm_only_text": llm_only_text,
                "rag_text": rag_text,
                "llm_only_seconds": round(llm_only_seconds, 3),
                "rag_seconds": round(rag_seconds, 3),
                "model": model_name,
            })
            print(
                f"{model_name} [{sample_number}/{len(detected_samples)}] "
                f"{sample['sample_id']}: complete",
                flush=True,
            )
    finally:
        unload_llm_model(session, model_name)

    return rows


def build_blind_files(result_rows, run_id):
    blind_items = []
    for row in result_rows:
        for method, text in (
            ("LLM-only", row["llm_only_text"]),
            ("LLM+RAG", row["rag_text"]),
        ):
            blind_items.append({
                "sample_id": row["sample_id"],
                "true_label": row["true_label"],
                "detected_label": row["detected_label"],
                "confidence": row["confidence"],
                "explanation_text": text,
                "method": method,
                "model": row["model"],
            })

    order = np.random.default_rng(BLIND_RANDOM_SEED).permutation(len(blind_items))
    blind_rows = []
    answer_key_rows = []
    for blind_number, item_index in enumerate(order, start=1):
        item = blind_items[int(item_index)]
        blind_id = f"B{blind_number:03d}"
        blind_rows.append({
            "blind_id": blind_id,
            "true_label": item["true_label"],
            "detected_label": item["detected_label"],
            "confidence": item["confidence"],
            "explanation_text": item["explanation_text"],
        })
        answer_key_rows.append({
            "blind_id": blind_id,
            "sample_id": item["sample_id"],
            "method": item["method"],
            "model": item["model"],
        })

    blind_path = OUTPUT_DIR / f"evaluation_blind_{run_id}.csv"
    answer_key_path = OUTPUT_DIR / f"evaluation_answer_key_{run_id}.csv"
    write_csv(blind_path, BLIND_FIELDS, blind_rows)
    write_csv(answer_key_path, ANSWER_KEY_FIELDS, answer_key_rows)
    return blind_path, answer_key_path


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    samples = get_eval_samples()
    detector = load_detector()
    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "y_test.npy", mmap_mode="r")
    detected_samples = detect_samples(samples, detector, x_test, y_test)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUT_DIR / f"evaluation_results_{run_id}.csv"
    result_rows = []

    with requests.Session() as session:
        for model_name in MODEL_NAMES:
            print(f"Loading {model_name}; processing all samples sequentially.", flush=True)
            result_rows.extend(
                evaluate_model(session, model_name, detected_samples)
            )

    write_csv(results_path, RESULT_FIELDS, result_rows)
    blind_path, answer_key_path = build_blind_files(result_rows, run_id)
    print(f"Results: {results_path}")
    print(f"Blind: {blind_path}")
    print(f"Answer key: {answer_key_path}")


if __name__ == "__main__":
    main()
