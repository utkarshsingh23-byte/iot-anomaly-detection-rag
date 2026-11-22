import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scipy.stats import wilcoxon


OUTPUT_FIELDS = (
    "section",
    "metric_label",
    "condition",
    "model",
    "metric",
    "n",
    "mean",
    "standard_deviation",
    "wilcoxon_statistic",
    "p_value",
    "notes",
)
IDENTIFIER_FIELDS = {"metric_label", "sample_id", "method", "model", "detected_label"}
LLM_METHODS = {"llm_only", "llm_rag"}
METRIC_GROUP_PREFIXES = (
    ("retrieval", "retrieval_"),
    ("relevance", "relevance_"),
    ("accuracy", "accuracy_"),
    ("actionability", "actionability_"),
    ("consistency", "consistency_"),
    ("clarity", "clarity_"),
)


def parse_number(value):
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None
    if text.lower() == "true":
        return 1.0
    if text.lower() == "false":
        return 0.0

    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def condition_name(method):
    return "llm_rag" if method == "rag" else method


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not have a CSV header.")
        return reader.fieldnames, list(reader)


def require_fields(path, fields, required):
    missing = sorted(set(required) - set(fields))
    if missing:
        raise ValueError(
            f"{path} is missing required CSV column(s): {', '.join(missing)}"
        )


def metric_columns(fields, rows):
    columns = []
    for field in fields:
        if field in IDENTIFIER_FIELDS:
            continue
        if any(parse_number(row.get(field)) is not None for row in rows):
            columns.append(field)
    return columns


def make_row(section, metric_label, condition="", model="", metric="", n=""):
    return {
        "section": section,
        "metric_label": metric_label,
        "condition": condition,
        "model": model,
        "metric": metric,
        "n": n,
        "mean": "",
        "standard_deviation": "",
        "wilcoxon_statistic": "",
        "p_value": "",
        "notes": "",
    }


def summarize_metrics(rows, columns):
    grouped_values = defaultdict(list)
    paired_values = defaultdict(dict)

    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise ValueError("Every metrics CSV row must have a non-empty sample_id.")

        condition = condition_name(row.get("method", "").strip())
        model = row.get("model", "").strip()
        for column in columns:
            value = parse_number(row.get(column))
            if value is not None:
                grouped_values[(condition, model, column)].append(value)

        if condition in LLM_METHODS:
            pair_key = (condition, model, sample_id)
            if pair_key in paired_values:
                raise ValueError(
                    "Duplicate metrics row for "
                    f"method={condition}, model={model}, sample_id={sample_id}."
                )
            paired_values[pair_key] = {
                column: parse_number(row.get(column)) for column in columns
            }

    llm_models = sorted(
        {
            row.get("model", "").strip()
            for row in rows
            if condition_name(row.get("method", "").strip()) in LLM_METHODS
        }
    )
    aggregate_groups = {("template", ""), ("retrieval_only", "")}
    for model in llm_models:
        aggregate_groups.add(("llm_only", model))
        aggregate_groups.add(("llm_rag", model))
    aggregate_groups.update(
        (condition, model) for condition, model, _ in grouped_values
    )

    output_rows = []
    for condition, model in sorted(aggregate_groups):
        for column in columns:
            values = grouped_values.get((condition, model, column), [])
            output = make_row(
                "metric_summary",
                "automated proxy metric",
                condition=condition,
                model=model,
                metric=column,
                n=len(values),
            )
            if values:
                output["mean"] = statistics.mean(values)
                if len(values) >= 2:
                    output["standard_deviation"] = statistics.stdev(values)
                else:
                    output["notes"] = "Standard deviation unavailable for n < 2."
            else:
                output["notes"] = "No numeric values available in the input."
            output_rows.append(output)

    models = sorted({model for _, model, _ in paired_values})
    for model in models:
        left_by_sample = {
            sample_id: values
            for (method, row_model, sample_id), values in paired_values.items()
            if method == "llm_only" and row_model == model
        }
        right_by_sample = {
            sample_id: values
            for (method, row_model, sample_id), values in paired_values.items()
            if method == "llm_rag" and row_model == model
        }

        for column in columns:
            left = []
            right = []
            for sample_id in sorted(left_by_sample.keys() & right_by_sample.keys()):
                left_value = left_by_sample[sample_id][column]
                right_value = right_by_sample[sample_id][column]
                if left_value is not None and right_value is not None:
                    left.append(left_value)
                    right.append(right_value)

            output = make_row(
                "wilcoxon",
                "automated proxy metric",
                condition="llm_only vs llm_rag",
                model=model,
                metric=column,
                n=len(left),
            )
            if len(left) < 2:
                output["notes"] = (
                    "Not tested: fewer than two paired observations have values "
                    "for this metric."
                )
            elif all(left_value == right_value for left_value, right_value in zip(left, right)):
                output["wilcoxon_statistic"] = 0.0
                output["p_value"] = 1.0
                output["notes"] = "All paired differences were zero."
            else:
                result = wilcoxon(
                    left,
                    right,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                output["wilcoxon_statistic"] = result.statistic
                output["p_value"] = result.pvalue
            output_rows.append(output)

    return output_rows


def summarize_generation_times(comparison_path, baseline_path):
    comparison_fields, comparison_rows = read_csv(comparison_path)
    require_fields(
        comparison_path,
        comparison_fields,
        ("llm_only_seconds", "rag_seconds"),
    )
    baseline_fields, baseline_rows = read_csv(baseline_path)
    require_fields(baseline_path, baseline_fields, ("method", "seconds"))

    grouped_values = defaultdict(list)
    for row in comparison_rows:
        model = row.get("model", "").strip()
        for source_column, condition in (
            ("llm_only_seconds", "llm_only"),
            ("rag_seconds", "llm_rag"),
        ):
            value = parse_number(row.get(source_column))
            if value is not None:
                grouped_values[(condition, model)].append(value)

    for row in baseline_rows:
        condition = row.get("method", "").strip()
        if condition not in {"template", "retrieval_only"}:
            continue
        value = parse_number(row.get("seconds"))
        if value is not None:
            grouped_values[(condition, row.get("model", "").strip())].append(value)

    output_rows = []
    for (condition, model), values in sorted(grouped_values.items()):
        output = make_row(
            "generation_time",
            "measured generation time",
            condition=condition,
            model=model,
            metric="seconds",
            n=len(values),
        )
        output["mean"] = statistics.mean(values)
        if len(values) >= 2:
            output["standard_deviation"] = statistics.stdev(values)
        else:
            output["notes"] = "Standard deviation unavailable for n < 2."
        output_rows.append(output)
    return output_rows


def write_charts(summary_rows, charts_dir):
    try:
        import matplotlib
    except ImportError:
        print("Charts skipped: matplotlib is not installed in this environment.")
        return []

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_rows = [
        row for row in summary_rows if row["section"] == "metric_summary"
    ]
    written_paths = []
    charts_dir.mkdir(parents=True, exist_ok=True)

    for group_name, prefix in METRIC_GROUP_PREFIXES:
        group_metrics = sorted(
            {
                row["metric"]
                for row in metric_rows
                if row["metric"].startswith(prefix)
            }
        )
        if not group_metrics:
            continue

        figure, axes = plt.subplots(
            len(group_metrics),
            1,
            figsize=(10, max(3.5, 3.2 * len(group_metrics))),
            squeeze=False,
        )
        for axis, metric in zip(axes[:, 0], group_metrics):
            values = [
                row
                for row in metric_rows
                if row["metric"] == metric and row["mean"] != ""
            ]
            labels = [
                f"{row['condition']}\n{row['model']}" if row["model"] else row["condition"]
                for row in values
            ]
            means = [float(row["mean"]) for row in values]
            deviations = [
                float(row["standard_deviation"])
                if row["standard_deviation"] != ""
                else 0.0
                for row in values
            ]
            axis.bar(range(len(values)), means, yerr=deviations, capsize=3)
            axis.set_title(metric)
            axis.set_xticks(range(len(labels)))
            axis.set_xticklabels(labels, rotation=25, ha="right")
            axis.set_ylabel("Mean")
            axis.grid(axis="y", alpha=0.25)

        figure.tight_layout()
        chart_path = charts_dir / f"{group_name}.png"
        figure.savefig(chart_path, dpi=150)
        plt.close(figure)
        written_paths.append(chart_path)

    return written_paths


def format_pair_counts(summary_rows):
    counts = defaultdict(int)
    for row in summary_rows:
        if row["section"] == "wilcoxon":
            counts[row["model"]] = max(counts[row["model"]], int(row["n"]))
    if not counts:
        return "no LLM-only/RAG model pairs were present"
    return ", ".join(
        f"{model or '(blank model)'}: n={count}"
        for model, count in sorted(counts.items())
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize explanation metrics and generation times."
    )
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--comparison-csv", required=True, type=Path)
    parser.add_argument("--baselines-csv", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--charts-dir", type=Path)
    args = parser.parse_args()

    metrics_fields, metrics_rows = read_csv(args.metrics_csv)
    require_fields(args.metrics_csv, metrics_fields, ("sample_id", "method", "model"))
    columns = metric_columns(metrics_fields, metrics_rows)
    if not columns:
        raise ValueError(f"{args.metrics_csv} contains no numeric explanation metrics.")

    summary_rows = summarize_metrics(metrics_rows, columns)
    summary_rows.extend(
        summarize_generation_times(args.comparison_csv, args.baselines_csv)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_csv or args.metrics_csv.with_name(
        f"evaluation_summary_{timestamp}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    charts_dir = args.charts_dir or output_path.with_name(
        f"{output_path.stem}_charts"
    )
    chart_paths = write_charts(summary_rows, charts_dir)

    print(f"Metrics input rows: {len(metrics_rows)}")
    print(f"Numeric explanation metrics: {len(columns)}")
    method_counts = defaultdict(int)
    for row in metrics_rows:
        method_counts[condition_name(row.get("method", "").strip())] += 1
    print(
        "Explanation rows by condition: "
        + ", ".join(
            f"{condition}={count}"
            for condition, count in sorted(method_counts.items())
        )
    )
    print(f"Summary rows written: {len(summary_rows)}")
    print(f"Summary CSV: {output_path.resolve()}")
    print(f"Paired observations available: {format_pair_counts(summary_rows)}")
    if chart_paths:
        print("Charts written:")
        for chart_path in chart_paths:
            print(f"  {chart_path.resolve()}")

    print(
        "\nNotes:\n"
        "- Explanation scores are automated proxy metrics, not human-verified accuracy.\n"
        "- NLI and embedding-similarity judgments can be wrong.\n"
        "- n=10 is small for a paired statistical comparison; the current input's "
        f"eligible pair counts are {format_pair_counts(summary_rows)}."
    )


if __name__ == "__main__":
    main()
