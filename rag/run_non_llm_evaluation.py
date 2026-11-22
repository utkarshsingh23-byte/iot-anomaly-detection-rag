import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.run_batch_evaluation import (
    DATA_DIR,
    OUTPUT_DIR,
    detect_samples,
    get_eval_samples,
    get_retrieval_query,
    load_detector,
    retrieve,
)


RESULT_FIELDS = [
    "sample_id",
    "method",
    "model",
    "text",
    "retrieved_sources",
    "seconds",
]
TEMPLATE_OPENERS = {
    "Normal": "The detector classified this window as Normal.",
    "DDoS": "The detector classified this window as DDoS.",
    "Botnet": "The detector classified this window as Botnet.",
}
TOP_FEATURE_COUNT = 5


def top_feature_values(features, limit=TOP_FEATURE_COUNT):
    largest_by_name = {}
    for timestep, row in enumerate(features, start=1):
        for name, value in row.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            magnitude = abs(numeric_value)
            current = largest_by_name.get(name)
            if current is None or magnitude > current[0]:
                largest_by_name[name] = (
                    magnitude,
                    timestep,
                    name,
                    numeric_value,
                )

    return sorted(
        largest_by_name.values(),
        key=lambda item: (-item[0], item[2]),
    )[:limit]


def build_template_text(detected_label, confidence, features):
    opener = TEMPLATE_OPENERS[detected_label]
    top_values = top_feature_values(features)
    if top_values:
        feature_text = "; ".join(
            f"{name} at timestep {timestep}: {value:.6g}"
            for _, timestep, name, value in top_values
        )
    else:
        feature_text = "no finite feature values were available"

    return (
        f"{opener} Confidence: {confidence:.1%}. "
        f"The highest-magnitude recorded feature values are: {feature_text}. "
        "These are observed measurements and do not establish a cause."
    )


def build_retrieval_only_text(detected_label, chunks):
    if detected_label == "Normal":
        return "Retrieval skipped because the detector classified this window as Normal."
    if not chunks:
        return "No reference chunks were retrieved."

    formatted_chunks = []
    for chunk in chunks:
        labels = []
        if chunk.get("source") is not None:
            labels.append(f"Source: {chunk['source']}")
        if chunk.get("page") is not None:
            labels.append(f"Page: {chunk['page']}")
        label = " | ".join(labels) if labels else "Reference"
        formatted_chunks.append(f"[{label}]\n{chunk['text']}")

    return "\n\n".join(formatted_chunks)


def serialize_sources(chunks):
    sources = []
    for chunk in chunks:
        source = {}
        for key in ("source", "page", "score"):
            if chunk.get(key) is not None:
                source[key] = chunk[key]
        sources.append(source)
    return json.dumps(sources, ensure_ascii=False)


def evaluate_non_llm_methods(detected_samples):
    rows = []
    for sample_number, (sample, detected_label, confidence) in enumerate(
        detected_samples,
        start=1,
    ):
        features = sample["features"]

        template_start = time.perf_counter()
        template_text = build_template_text(
            detected_label,
            confidence,
            features,
        )
        template_seconds = time.perf_counter() - template_start
        rows.append({
            "sample_id": sample["sample_id"],
            "method": "template",
            "model": "",
            "text": template_text,
            "retrieved_sources": "[]",
            "seconds": round(template_seconds, 3),
        })

        retrieval_start = time.perf_counter()
        if detected_label == "Normal":
            chunks = []
        else:
            chunks = retrieve(
                get_retrieval_query(detected_label, features),
                k=3,
                attack_type=detected_label,
            )
        retrieval_text = build_retrieval_only_text(detected_label, chunks)
        retrieval_seconds = time.perf_counter() - retrieval_start
        rows.append({
            "sample_id": sample["sample_id"],
            "method": "retrieval_only",
            "model": "",
            "text": retrieval_text,
            "retrieved_sources": serialize_sources(chunks),
            "seconds": round(retrieval_seconds, 3),
        })

        print(
            f"[{sample_number}/{len(detected_samples)}] "
            f"{sample['sample_id']}: template and retrieval_only complete",
            flush=True,
        )

    return rows


def write_results(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    samples = get_eval_samples()
    detector = load_detector()
    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "y_test.npy", mmap_mode="r")
    detected_samples = detect_samples(samples, detector, x_test, y_test)
    result_rows = evaluate_non_llm_methods(detected_samples)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUT_DIR / f"evaluation_non_llm_{run_id}.csv"
    write_results(results_path, result_rows)
    print(f"Results: {results_path}")
    print(f"Rows: {len(result_rows)}")


if __name__ == "__main__":
    main()
