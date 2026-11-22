import json
import re
import sys

import requests


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
SUPPORTED_MODELS = ("llama3.2:3b", "gemma2:2b", "phi3:mini")
PHI3_MODEL = "phi3:mini"
_OPTIONS = {
    "temperature": 0.2,
    "num_predict": 280,
    "top_p": 0.85,
    "repeat_penalty": 1.15,
}
_PHI3_HEADINGS = (
    "THREAT ASSESSMENT",
    "OBSERVED FEATURES",
    "IMMEDIATE ACTIONS",
    "RECOMMENDATIONS",
    "EVIDENCE",
)
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_MALFORMED_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d*)?[A-Za-z]\w*"
)
_REFERENCE_PATTERN = re.compile(r"\[?\bReference\s+(\d+)\b[^\]]*\]?", re.IGNORECASE)
_TECHNIQUE_PATTERN = re.compile(r"\b(?:T\d{4}(?:[._]\d{3})?|AN\d{4,})\b", re.IGNORECASE)


def _format_features(features):
    return json.dumps(features, indent=2, ensure_ascii=False, default=str)


def _phi3_format_instructions(prompt):
    return (
        f"{prompt}\n\n"
        "Phi-3 output requirements: use exactly these headings, in this order: "
        "THREAT ASSESSMENT, OBSERVED FEATURES, IMMEDIATE ACTIONS, "
        "RECOMMENDATIONS, EVIDENCE. In OBSERVED FEATURES, copy one to five "
        "feature names and numeric values exactly from the input; do not add "
        "units. Do not invent references or technical identifiers. Keep the "
        "answer concise and finish every sentence."
    )


def build_llm_only_prompt(attack_type, features, model=DEFAULT_MODEL):
    prompt = (
        "You are explaining an IoT anomaly detector result.\n"
        f"Detector prediction: {attack_type}\n"
        "Observed traffic feature values:\n"
        f"{_format_features(features)}\n\n"
        "Explain what is directly observable in the supplied feature values, "
        "separating observations from interpretations. Do not invent feature "
        "values or assume facts not provided. Treat the detector prediction as "
        "fixed; do not change or relabel it. State when the supplied information "
        "is insufficient to support a conclusion."
    )
    return _phi3_format_instructions(prompt) if model == PHI3_MODEL else prompt


def build_rag_prompt(
    attack_type, features, retrieved_chunks, model=DEFAULT_MODEL
):
    references = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        source_label = f"[Reference {index} | Source: {chunk['source']}"
        if chunk.get("page") is not None:
            source_label += f" | Page: {chunk['page']}"
        references.append(f"{source_label}]\n{chunk['text']}")

    reference_text = (
        "\n\n".join(references)
        if references
        else "No reference chunks were retrieved."
    )
    prompt = (
        "You are explaining an IoT anomaly detector result using the supplied "
        "traffic features and reference material.\n"
        f"Detector prediction: {attack_type}\n"
        "Observed traffic feature values:\n"
        f"{_format_features(features)}\n\n"
        "Reference material:\n"
        f"{reference_text}\n\n"
        "Separate observed traffic features from inferred behaviour. Use only "
        "the supplied references for cybersecurity-specific claims and "
        "recommendations. Do not invent citations or attribute claims to a "
        "reference unless it supports them. If the references are insufficient, "
        "say so explicitly. Treat the detector prediction as fixed; never change, "
        "relabel, or contradict it."
    )
    return _phi3_format_instructions(prompt) if model == PHI3_MODEL else prompt


def _feature_rows(features):
    if isinstance(features, dict):
        return [features]
    if isinstance(features, (list, tuple)):
        return [row for row in features if isinstance(row, dict)]
    return []


def _decimal_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if isinstance(value, int) else float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _phi3_validation_errors(text, attack_type, features, retrieved_chunks):
    errors = []
    if not isinstance(text, str) or not text.strip():
        return ["empty response"]

    heading_positions = [text.upper().find(heading) for heading in _PHI3_HEADINGS]
    if any(position < 0 for position in heading_positions):
        errors.append("required section heading missing")
    elif heading_positions != sorted(heading_positions):
        errors.append("section headings are out of order")

    words = re.findall(r"\b[\w'-]+\b", text)
    if len(words) > 220:
        errors.append("response is too long")
    completed_text = re.sub(r"[\s*_`\"'’)\]}]+$", "", text)
    if not completed_text or completed_text[-1] not in ".!?":
        errors.append("response appears truncated")

    label = str(attack_type).strip().casefold()
    other_labels = {
        candidate.casefold()
        for candidate in ("Normal", "DDoS", "Botnet")
        if candidate.casefold() != label
    }
    if any(
        re.search(rf"\b{re.escape(other_label)}\b", text, re.IGNORECASE)
        for other_label in other_labels
    ):
        errors.append("response mentions another detector class")

    retrieved_chunks = retrieved_chunks or []
    for reference_match in _REFERENCE_PATTERN.finditer(text):
        reference_number = int(reference_match.group(1))
        if not 1 <= reference_number <= len(retrieved_chunks):
            errors.append("response cites a reference that was not retrieved")
            break

    available_identifiers = set()
    for chunk in retrieved_chunks:
        content = f"{chunk.get('source', '')} {chunk.get('text', '')}"
        available_identifiers.update(
            identifier.upper().replace("_", ".")
            for identifier in _TECHNIQUE_PATTERN.findall(content)
        )
    for identifier in _TECHNIQUE_PATTERN.findall(text):
        if identifier.upper().replace("_", ".") not in available_identifiers:
            errors.append("response includes an identifier absent from retrieved sources")
            break

    rows = _feature_rows(features)
    feature_values = {}
    allowed_numbers = set()
    for row in rows:
        for name, value in row.items():
            numeric_value = _decimal_value(value)
            if numeric_value is None:
                continue
            feature_values.setdefault(str(name).casefold(), set()).add(numeric_value)
            allowed_numbers.add(numeric_value)

    observed_start = heading_positions[1] + len(_PHI3_HEADINGS[1]) if heading_positions[1] >= 0 else -1
    observed_end = heading_positions[2] if heading_positions[2] >= 0 else len(text)
    observed_text = text[observed_start:observed_end] if observed_start >= 0 else ""
    matched_features = 0
    for name, values in feature_values.items():
        match = re.search(
            rf"(?<![\w]){re.escape(name)}(?:['\"])?\s*[:=]\s*['\"]?"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            observed_text,
            re.IGNORECASE,
        )
        if not match:
            continue
        matched_features += 1
        if _decimal_value(float(match.group(1))) not in values:
            errors.append("an observed feature value does not match the input")
            break
    if not matched_features:
        errors.append("no exact feature name/value pair found in observed features")

    numeric_text = _REFERENCE_PATTERN.sub(" ", text)
    numeric_text = _TECHNIQUE_PATTERN.sub(" ", numeric_text)
    numeric_text = re.sub(r"(?m)^\s*\d+[.)]\s+", " ", numeric_text)
    if _MALFORMED_NUMBER_PATTERN.search(numeric_text):
        errors.append("response contains a malformed numeric value")
    else:
        for number_match in _NUMBER_PATTERN.finditer(numeric_text):
            if _decimal_value(float(number_match.group())) not in allowed_numbers:
                errors.append("response contains a numeric value absent from the input")
                break

    return list(dict.fromkeys(errors))


def _safe_phi3_fallback(attack_type, features, retrieved_chunks):
    rows = _feature_rows(features)
    if rows:
        latest = rows[-1]
        sample = dict(list(latest.items())[:5])
        sampled_field_count = len(sample)
        observed = (
            f"Latest-timestep sample (first {sampled_field_count} supplied fields; "
            "not ranked): "
            f"`{json.dumps(sample, ensure_ascii=False, default=str)}`"
        )
    else:
        observed = "No structured feature values were available."

    label = str(attack_type).strip()
    if label.casefold() == "ddos":
        actions = (
            "Verify the alert against packet captures and the recent traffic baseline "
            "before attributing a cause.",
            "Use existing filtering or rate-limiting controls only if service impact "
            "is corroborated.",
        )
        recommendation = "Monitor traffic and service health for sustained deviations."
    elif label.casefold() == "botnet":
        actions = (
            "Review the affected device's telemetry and outbound connections before "
            "classifying it as compromised.",
            "Follow the existing isolation procedure only after the alert is "
            "corroborated.",
        )
        recommendation = "Monitor device behavior and investigate verified indicators."
    else:
        actions = (
            "Compare the detector result with recent baseline traffic and relevant "
            "device logs.",
            "Escalate only if independent telemetry confirms an ongoing issue.",
        )
        recommendation = "Continue monitoring for persistent deviations."

    evidence = (
        f"{len(retrieved_chunks)} retrieved reference chunk(s) are available for "
        "review; this fallback does not claim they confirm this flow."
        if retrieved_chunks
        else "No retrieved references were used."
    )
    return (
        "**Phi-3 response failed validation; showing a safe fallback.**\n\n"
        f"**THREAT ASSESSMENT**\n"
        f"The detector predicted {label}. This is a model result, not independent "
        "proof of the cause.\n\n"
        f"**OBSERVED FEATURES**\n{observed}\n\n"
        f"**IMMEDIATE ACTIONS**\n"
        f"1. {actions[0]}\n"
        f"2. {actions[1]}\n\n"
        f"**RECOMMENDATIONS**\n{recommendation}\n\n"
        f"**EVIDENCE**\n{evidence}"
    )


def finalize_explanation(text, attack_type, features, model=DEFAULT_MODEL,
                         retrieved_chunks=None):
    if model != PHI3_MODEL or (
        isinstance(text, str)
        and any(
            marker in text
            for marker in (
                "Cannot reach Ollama.",
                "Ollama timed out",
                "LLM error:",
            )
        )
    ):
        return text

    errors = _phi3_validation_errors(
        text,
        attack_type,
        features,
        retrieved_chunks,
    )
    if not errors:
        return text

    print(
        "Phi-3 response failed validation "
        f"({'; '.join(errors)}); displaying the safe fallback.",
        file=sys.stderr,
        flush=True,
    )
    return _safe_phi3_fallback(
        attack_type,
        features,
        retrieved_chunks or [],
    )


def _retrieve(query, attack_type):
    if __package__:
        from .retrieve import retrieve
    else:
        from retrieve import retrieve

    return retrieve(query, k=3, attack_type=attack_type)


def stream_explanation(
    attack_type,
    features,
    use_rag=True,
    model=DEFAULT_MODEL,
    retrieved_chunks=None,
):
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported Ollama model {model!r}. "
            f"Choose one of: {', '.join(SUPPORTED_MODELS)}."
        )

    if retrieved_chunks is None:
        retrieved_chunks = []

    normal_detection = str(attack_type).strip().lower() == "normal"
    if use_rag and not normal_detection:
        if not retrieved_chunks:
            retrieval_query = (
                f"IoT network {attack_type} anomaly; observed features: "
                f"{_format_features(features)}"
            )
            retrieved_chunks.extend(_retrieve(retrieval_query, attack_type))

        if retrieved_chunks:
            prompt = build_rag_prompt(
                attack_type, features, retrieved_chunks, model=model
            )
        else:
            prompt = build_llm_only_prompt(attack_type, features, model=model)
    else:
        prompt = build_llm_only_prompt(attack_type, features, model=model)

    try:
        with requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": _OPTIONS,
            },
            stream=True,
            timeout=90,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield (
            "\n⚠️  Cannot reach Ollama.\n\n"
            "Start it:   ollama serve\n"
            f"Pull model: ollama pull {model}\n\n"
            "LLM will auto-retry on the next detected attack."
        )
    except requests.exceptions.Timeout:
        yield "\n⚠️  Ollama timed out (90s). The model may be loading — next attack will retry."
    except Exception as error:
        yield f"\n⚠️  LLM error: {type(error).__name__}: {error}"


if __name__ == "__main__":
    example_attack_type = "DDoS"
    example_features = {
        "syn_count": 12,
        "Rate": 300.0,
        "Srate": 285.0,
    }

    for label, rag_enabled in (("LLM-only", False), ("LLM + RAG", True)):
        print(f"\n{label} explanation:")
        for output_token in stream_explanation(
            example_attack_type,
            example_features,
            use_rag=rag_enabled,
        ):
            print(output_token, end="", flush=True)
        print()
