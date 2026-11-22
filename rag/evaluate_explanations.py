import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT_DIR / "rag"
CHUNKS_PATH = RAG_DIR / "rag_store" / "chunks.json"
MITRE_DIR = ROOT_DIR / "data" / "rag_documents" / "mitre_attack"
CISA_DIR = ROOT_DIR / "data" / "rag_documents" / "cisa"
CSK_DIR = ROOT_DIR / "data" / "rag_documents" / "csk"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_CONFIDENCE_THRESHOLD = 0.70
FALLBACK_COSINE_THRESHOLD = 0.55
NLI_CANDIDATE_COUNT = 3
NUMERIC_REL_TOLERANCE = 1e-3
NUMERIC_ABS_TOLERANCE = 1e-6
FIXED_SEED = 42

METHOD_FIELDS = {
    "llm_only": "llm_only_text",
    "rag": "rag_text",
}
BASELINE_METHODS = ("template", "retrieval_only")
CLASS_PATTERNS = {
    "Normal": (r"\bnormal\b",),
    "DDoS": (
        r"\bddos\b",
        r"\bdistributed\s+denial[-\s]+of[-\s]+service\b",
    ),
    "Botnet": (r"\bbot[\s-]?nets?\b",),
}
DIAGNOSIS_CUES = (
    "classified",
    "diagnosed",
    "identified",
    "detected",
    "predicted",
    "labeled",
    "labelled",
)
TECHNIQUE_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
NUMBER_PATTERN = (
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?"
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*#(\[])")
LIST_MARKER = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
SOURCE_FILENAME = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:pdf|txt)\b",
    re.IGNORECASE,
)

MITIGATION_TERMS = (
    "allowlist",
    "apply",
    "backup",
    "block",
    "blocklist",
    "configure",
    "contain",
    "deny",
    "deploy",
    "detect",
    "disable",
    "drop",
    "enforce",
    "filter",
    "firewall",
    "harden",
    "implement",
    "isolate",
    "limit",
    "monitor",
    "patch",
    "protect",
    "quarantine",
    "rate limit",
    "redirect",
    "reset",
    "restrict",
    "restore",
    "revoke",
    "review",
    "rotate",
    "scrub",
    "segment",
    "sinkhole",
    "update",
    "upgrade",
    "validate",
    "mitigate",
    "incident response",
    "access control",
    "traffic filtering",
)
ACTION_VERBS = (
    "apply",
    "block",
    "configure",
    "contain",
    "deny",
    "deploy",
    "disable",
    "drop",
    "enable",
    "enforce",
    "filter",
    "harden",
    "implement",
    "inspect",
    "isolate",
    "limit",
    "mitigate",
    "monitor",
    "patch",
    "protect",
    "quarantine",
    "redirect",
    "reset",
    "restrict",
    "restore",
    "revoke",
    "rotate",
    "scrub",
    "segment",
    "sinkhole",
    "update",
    "upgrade",
    "validate",
)

OUTPUT_FIELDS = [
    "metric_label",
    "sample_id",
    "method",
    "model",
    "detected_label",
    "retrieval_attack_type_match_rate",
    "retrieval_mean_similarity",
    "retrieval_source_coverage",
    "retrieval_cited_source_names",
    "retrieval_citation_count",
    "retrieval_citation_validity",
    "relevance_cosine",
    "relevance_detected_class_named",
    "accuracy_backend",
    "accuracy_sentence_count",
    "accuracy_supported_share",
    "accuracy_unsupported_share",
    "accuracy_contradicted_share",
    "accuracy_technique_ids",
    "accuracy_technique_id_match_rate",
    "accuracy_invalid_technique_ids",
    "actionability_concrete_action_count",
    "actionability_step_count",
    "actionability_action_step_share",
    "actionability_required_section_presence",
    "actionability_required_section_share",
    "actionability_mitigation_vocabulary_matches",
    "consistency_detected_class_named",
    "consistency_other_class_diagnosis_absent",
    "consistency_numeric_feature_values_checked",
    "consistency_numeric_feature_values_match_share",
    "clarity_flesch_reading_ease",
    "clarity_word_count",
    "clarity_repeated_trigram_rate",
    "clarity_format_compliant",
    "quality_relevance",
    "quality_accuracy",
    "quality_actionability",
    "quality_consistency",
    "quality_clarity",
]
QUALITY_SCORE_NAMES = (
    "relevance",
    "accuracy",
    "actionability",
    "consistency",
    "clarity",
)


def read_csv_rows(path, required_fields, label):
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{label} CSV has no header: {path}")
        missing = sorted(set(required_fields) - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"{label} CSV is missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{label} CSV has no data rows: {path}")
    return rows


def parse_retrieved_sources(value, context):
    if not value:
        return []
    try:
        sources = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid retrieved_sources JSON in {context}") from error
    if not isinstance(sources, list) or any(
        not isinstance(source, dict) for source in sources
    ):
        raise ValueError(f"retrieved_sources must be a JSON list of objects in {context}")
    return sources


def normalize_label(value, context):
    label = value.strip()
    if label not in CLASS_PATTERNS:
        raise ValueError(f"Unknown detected_label {value!r} in {context}")
    return label


def load_inputs(comparison_path, baseline_path):
    comparison_rows = read_csv_rows(
        comparison_path,
        (
            "sample_id",
            "detected_label",
            "retrieved_sources",
            "llm_only_text",
            "rag_text",
        ),
        "comparison",
    )
    baseline_rows = read_csv_rows(
        baseline_path,
        ("sample_id", "method", "text", "retrieved_sources"),
        "baseline",
    )

    comparison_by_sample = defaultdict(list)
    detected_by_sample = {}
    source_by_sample = {}
    for row_number, row in enumerate(comparison_rows, start=2):
        context = f"{comparison_path.name}:{row_number}"
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ValueError(f"Empty sample_id in {context}")
        label = normalize_label(row["detected_label"], context)
        previous_label = detected_by_sample.setdefault(sample_id, label)
        if previous_label != label:
            raise ValueError(f"Inconsistent detected_label for {sample_id}")

        retrieved = parse_retrieved_sources(row["retrieved_sources"], context)
        previous_sources = source_by_sample.setdefault(sample_id, retrieved)
        if previous_sources != retrieved:
            raise ValueError(f"Inconsistent retrieved_sources for {sample_id}")
        comparison_by_sample[sample_id].append(row)

    baselines_by_sample = defaultdict(dict)
    for row_number, row in enumerate(baseline_rows, start=2):
        context = f"{baseline_path.name}:{row_number}"
        sample_id = row["sample_id"].strip()
        method = row["method"].strip()
        if method not in BASELINE_METHODS:
            raise ValueError(f"Unsupported baseline method {method!r} in {context}")
        if sample_id not in detected_by_sample:
            raise ValueError(f"Baseline sample {sample_id!r} is absent from comparison CSV")
        if method in baselines_by_sample[sample_id]:
            raise ValueError(f"Duplicate {method} baseline row for {sample_id}")
        row["_retrieved_sources"] = parse_retrieved_sources(
            row["retrieved_sources"],
            context,
        )
        baselines_by_sample[sample_id][method] = row

    if set(comparison_by_sample) != set(baselines_by_sample):
        raise ValueError("Comparison and baseline CSVs contain different sample IDs")
    for sample_id, methods in baselines_by_sample.items():
        if set(methods) != set(BASELINE_METHODS):
            raise ValueError(
                f"Expected template and retrieval_only rows for {sample_id}"
            )

    return comparison_by_sample, baselines_by_sample, detected_by_sample


def get_top_features(features, limit=5):
    highest_by_name = {}
    for timestep, values in enumerate(features, start=1):
        for name, value in values.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            magnitude = abs(numeric_value)
            previous = highest_by_name.get(name)
            if previous is None or magnitude > previous[0]:
                highest_by_name[name] = (
                    magnitude,
                    name,
                    numeric_value,
                    timestep,
                )
    return sorted(
        highest_by_name.values(),
        key=lambda item: (-item[0], item[1]),
    )[:limit]


def feature_description(detected_label, features):
    top = get_top_features(features)
    rendered = "; ".join(
        f"{name}={value:.6g} (timestep {timestep})"
        for _, name, value, timestep in top
    )
    return f"Detected class: {detected_label}. Top observed input features: {rendered}."


def make_explanations(comparison_by_sample, baselines_by_sample, detected_by_sample,
                      feature_by_sample, limit=None):
    explanations = []
    for sample_id, comparison_rows in comparison_by_sample.items():
        features = feature_by_sample.get(sample_id)
        if features is None:
            raise ValueError(
                f"Sample {sample_id!r} is not in the fixed evaluation sample set"
            )
        detected_label = detected_by_sample[sample_id]
        inserted_baselines = False

        for comparison_row in comparison_rows:
            model = comparison_row.get("model", "").strip()
            retrieved = parse_retrieved_sources(
                comparison_row["retrieved_sources"],
                f"comparison row for {sample_id}",
            )
            for method, text_field in METHOD_FIELDS.items():
                explanations.append({
                    "sample_id": sample_id,
                    "method": method,
                    "model": model,
                    "text": comparison_row[text_field] or "",
                    "detected_label": detected_label,
                    "features": features,
                    "retrieved_sources": retrieved if method == "rag" else [],
                })

            if not inserted_baselines:
                for method in BASELINE_METHODS:
                    baseline = baselines_by_sample[sample_id][method]
                    explanations.append({
                        "sample_id": sample_id,
                        "method": method,
                        "model": "",
                        "text": baseline["text"] or "",
                        "detected_label": detected_label,
                        "features": features,
                        "retrieved_sources": baseline["_retrieved_sources"],
                    })
                inserted_baselines = True

    if limit is not None:
        explanations = explanations[:limit]
    return explanations


def load_chunks_and_sources():
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Indexed chunks not found: {CHUNKS_PATH}")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"No indexed chunks found in {CHUNKS_PATH}")

    cisa_sources = {path.name for path in CISA_DIR.glob("*.pdf")}
    csk_sources = {path.name for path in CSK_DIR.glob("*.txt")}
    mitre_files = sorted(MITRE_DIR.glob("*.txt"))
    if not mitre_files:
        raise FileNotFoundError(f"No MITRE text documents found in {MITRE_DIR}")

    technique_ids = set()
    for path in mitre_files:
        document = path.read_text(encoding="utf-8", errors="ignore")
        technique_ids.update(
            match.group(0).upper()
            for match in TECHNIQUE_PATTERN.finditer(document)
        )

    mitigation_sources = cisa_sources | csk_sources
    mitigation_corpus = " ".join(
        chunk["text"]
        for chunk in chunks
        if chunk.get("source") in mitigation_sources
    )
    mitigation_patterns = build_term_patterns(
        MITIGATION_TERMS,
        mitigation_corpus,
    )
    action_patterns = build_term_patterns(
        ACTION_VERBS,
        mitigation_corpus,
    )

    source_names = sorted({
        chunk["source"]
        for chunk in chunks
        if isinstance(chunk.get("source"), str)
    })
    source_type_map = defaultdict(set)
    for chunk in chunks:
        source = chunk.get("source")
        if not source:
            continue
        source_type_map[(source, chunk.get("page"))].add(chunk.get("attack_type"))
        source_type_map[(source, None)].add(chunk.get("attack_type"))

    return (
        chunks,
        source_names,
        source_type_map,
        technique_ids,
        mitigation_patterns,
        action_patterns,
    )


def build_term_patterns(terms, corpus):
    patterns = {}
    for term in terms:
        parts = [re.escape(part) for part in term.split()]
        expression = r"[-\s]+".join(parts)
        pattern = re.compile(
            rf"(?<![A-Za-z]){expression}[A-Za-z]*(?![A-Za-z])",
            re.IGNORECASE,
        )
        if pattern.search(corpus):
            patterns[term] = pattern
    return patterns


def load_models():
    from huggingface_hub.errors import CorruptedCacheException, LocalEntryNotFoundError
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True,
    )
    try:
        from sentence_transformers import CrossEncoder

        nli_model = CrossEncoder(
            NLI_MODEL,
            max_length=256,
            local_files_only=True,
        )
    except (
        CorruptedCacheException,
        ImportError,
        LocalEntryNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"WARNING: local NLI model unavailable ({type(error).__name__}: {error}); "
            "using embedding-similarity fallback. Contradiction share will be blank.",
            file=sys.stderr,
        )
        nli_model = None

    return embedding_model, nli_model


def split_sentences(text):
    sentences = []
    for line in text.splitlines():
        line = LIST_MARKER.sub("", line).strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"\[(?:Source|Reference)[^\]]*\]", line, re.IGNORECASE):
            continue
        line = re.sub(r"^\*{1,2}|\*{1,2}$", "", line).strip()
        sentences.extend(
            part.strip()
            for part in SENTENCE_BOUNDARY.split(line)
            if part.strip()
        )
    return sentences


def class_is_named(text, label):
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in CLASS_PATTERNS[label]
    )


def other_class_diagnosis_present(text, detected_label):
    diagnostic_class_labels = (
        label for label in CLASS_PATTERNS if label != detected_label
    )
    for label in diagnostic_class_labels:
        for class_pattern in CLASS_PATTERNS[label]:
            for cue in DIAGNOSIS_CUES:
                pattern = (
                    rf"\b{cue}\b\s+(?:as\s+)?(?:the\s+)?(?:an?\s+)?"
                    rf"(?:{class_pattern})"
                )
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            pattern = (
                rf"\b(?:is|was|appears to be|represents)\s+(?:the\s+)?"
                rf"(?:an?\s+)?(?:{class_pattern})\b"
            )
            if re.search(pattern, text, re.IGNORECASE):
                return True
    return False


def extract_numeric_feature_values(text, features):
    matches = []
    for feature_name in features[0]:
        name = re.escape(feature_name)
        prefix_pattern = re.compile(
            rf"(?<!\w){name}(?!\w)"
            rf"(?:\s+at\s+timestep\s+\d+)?"
            rf"\s*[`\"'*]*\s*(?::|=|\bis\b|\bwas\b)\s*[`\"']*"
            rf"({NUMBER_PATTERN})(?![\w]|\.\d)",
            re.IGNORECASE,
        )
        reverse_pattern = re.compile(
            rf"(?<![\w.])({NUMBER_PATTERN})\s+"
            rf"(?:for|in|of)\s+[`\"']?{name}(?!\w)",
            re.IGNORECASE,
        )
        matches.extend(
            (feature_name, float(value.replace(",", "")))
            for value in prefix_pattern.findall(text)
        )
        matches.extend(
            (feature_name, float(value.replace(",", "")))
            for value in reverse_pattern.findall(text)
        )
    return matches


def numeric_feature_match_share(text, features):
    claims = extract_numeric_feature_values(text, features)
    if not claims:
        return 0, None

    correct = 0
    values_by_name = defaultdict(list)
    for timestep in features:
        for name, value in timestep.items():
            values_by_name[name].append(float(value))

    for name, claimed_value in claims:
        if any(
            math.isclose(
                claimed_value,
                actual_value,
                rel_tol=NUMERIC_REL_TOLERANCE,
                abs_tol=NUMERIC_ABS_TOLERANCE,
            )
            for actual_value in values_by_name[name]
        ):
            correct += 1
    return len(claims), correct / len(claims)


def cited_source_names(text, known_sources):
    lowered_text = text.casefold()
    found = {
        source
        for source in known_sources
        if source.casefold() in lowered_text
    }
    known_lower = {source.casefold() for source in known_sources}
    for match in SOURCE_FILENAME.finditer(text):
        name = match.group(0)
        if name.casefold() not in known_lower:
            found.add(name)
    return sorted(found, key=str.casefold)


def retrieval_metrics(explanation, source_names, source_type_map):
    if explanation["method"] not in ("rag", "retrieval_only"):
        return {
            "retrieval_attack_type_match_rate": None,
            "retrieval_mean_similarity": None,
            "retrieval_source_coverage": None,
            "retrieval_cited_source_names": None,
            "retrieval_citation_count": None,
            "retrieval_citation_validity": None,
        }

    retrieved = explanation["retrieved_sources"]
    label = explanation["detected_label"]
    matches = 0
    scores = []
    retrieved_names = set()
    for item in retrieved:
        source = item.get("source")
        page = item.get("page")
        if source:
            retrieved_names.add(source)
            known_types = source_type_map.get((source, page), set())
            if not known_types and page is not None:
                known_types = source_type_map.get((source, None), set())
            if label in known_types or "General" in known_types:
                matches += 1
        if item.get("score") is not None:
            scores.append(float(item["score"]))

    cited = cited_source_names(explanation["text"], source_names)
    citation_validity = (
        sum(name in retrieved_names for name in cited) / len(cited)
        if cited else None
    )
    return {
        "retrieval_attack_type_match_rate": (
            matches / len(retrieved) if retrieved else None
        ),
        "retrieval_mean_similarity": (
            sum(scores) / len(scores) if scores else None
        ),
        "retrieval_source_coverage": len(retrieved_names),
        "retrieval_cited_source_names": json.dumps(cited, ensure_ascii=False),
        "retrieval_citation_count": len(cited),
        "retrieval_citation_validity": citation_validity,
    }


def word_tokens(text):
    return [word.lower() for word in WORD_PATTERN.findall(text)]


def syllable_count(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 1
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def clarity_metrics(text, sentences):
    words = word_tokens(text)
    word_count = len(words)
    if word_count and sentences:
        flesch = (
            206.835
            - 1.015 * (word_count / len(sentences))
            - 84.6 * (sum(syllable_count(word) for word in words) / word_count)
        )
    else:
        flesch = None

    trigrams = [
        tuple(words[index:index + 3])
        for index in range(max(0, len(words) - 2))
    ]
    repeated_trigram_rate = (
        (len(trigrams) - len(set(trigrams))) / len(trigrams)
        if trigrams else 0.0
    )
    markdown_balanced = (
        text.count("```") % 2 == 0
        and text.count("**") % 2 == 0
    )
    return {
        "clarity_flesch_reading_ease": flesch,
        "clarity_word_count": word_count,
        "clarity_repeated_trigram_rate": repeated_trigram_rate,
        "clarity_format_compliant": bool(text.strip()) and markdown_balanced,
    }


def extract_steps(text, action_patterns):
    list_items = []
    for line in text.splitlines():
        if LIST_MARKER.match(line):
            list_items.append(LIST_MARKER.sub("", line).strip())
    if list_items:
        return list_items

    return [
        sentence
        for sentence in split_sentences(text)
        if any(pattern.search(sentence) for pattern in action_patterns.values())
    ]


def starts_with_action_verb(text, action_patterns):
    cleaned = LIST_MARKER.sub("", text).strip()
    cleaned = re.sub(r"^[`*_]+", "", cleaned)
    return any(
        pattern.match(cleaned)
        for pattern in action_patterns.values()
    )


def actionability_metrics(text, features, class_named, mitigation_patterns,
                          action_patterns):
    mitigation_matches = sorted(
        term
        for term, pattern in mitigation_patterns.items()
        if pattern.search(text)
    )
    steps = extract_steps(text, action_patterns)
    action_count = sum(
        any(pattern.search(step) for pattern in action_patterns.values())
        for step in steps
    )
    action_starts = sum(
        starts_with_action_verb(step, action_patterns)
        for step in steps
    )
    mentioned_features = any(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)
        for name in features[0]
    )
    sections = {
        "classification": class_named,
        "observed_evidence": mentioned_features,
        "defensive_actions": action_count > 0,
    }
    return {
        "actionability_concrete_action_count": action_count,
        "actionability_step_count": len(steps),
        "actionability_action_step_share": (
            action_starts / len(steps) if steps else None
        ),
        "actionability_required_section_presence": json.dumps(
            sections,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "actionability_required_section_share": (
            sum(sections.values()) / len(sections)
        ),
        "actionability_mitigation_vocabulary_matches": json.dumps(
            mitigation_matches,
            ensure_ascii=False,
        ),
    }


def technique_metrics(text, known_technique_ids):
    mentioned = sorted({
        match.group(0).upper()
        for match in TECHNIQUE_PATTERN.finditer(text)
    })
    invalid = [technique for technique in mentioned if technique not in known_technique_ids]
    return {
        "accuracy_technique_ids": json.dumps(mentioned),
        "accuracy_technique_id_match_rate": (
            (len(mentioned) - len(invalid)) / len(mentioned)
            if mentioned else None
        ),
        "accuracy_invalid_technique_ids": json.dumps(invalid),
    }


def calculate_accuracy(sentences, sentence_embeddings, chunk_embeddings,
                       chunk_texts, nli_model, nli_labels):
    sentence_count = len(sentences)
    if not sentence_count:
        return {
            "accuracy_backend": (
                "nli_cross_encoder" if nli_model is not None
                else "embedding_similarity_fallback"
            ),
            "accuracy_sentence_count": 0,
            "accuracy_supported_share": None,
            "accuracy_unsupported_share": None,
            "accuracy_contradicted_share": None,
        }

    categories = []
    if nli_model is None:
        similarities = sentence_embeddings @ chunk_embeddings.T
        categories = [
            "supported"
            if float(similarity.max()) >= FALLBACK_COSINE_THRESHOLD
            else "unsupported"
            for similarity in similarities
        ]
        contradicted_share = None
        backend = "embedding_similarity_fallback"
    else:
        pair_locations = []
        pairs = []
        for sentence_index, sentence_vector in enumerate(sentence_embeddings):
            similarities = sentence_vector @ chunk_embeddings.T
            candidate_count = min(NLI_CANDIDATE_COUNT, len(similarities))
            candidate_indices = np.argsort(similarities)[-candidate_count:][::-1]
            for chunk_index in candidate_indices:
                pairs.append((chunk_texts[chunk_index], sentences[sentence_index]))
                pair_locations.append(sentence_index)

        predictions = np.asarray(
            nli_model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
                apply_softmax=True,
            )
        )
        if predictions.ndim != 2:
            raise ValueError("NLI cross-encoder did not return class scores")
        scores_by_sentence = defaultdict(list)
        for sentence_index, scores in zip(pair_locations, predictions):
            scores_by_sentence[sentence_index].append(scores)

        entailment_index = nli_labels["entailment"]
        contradiction_index = nli_labels["contradiction"]
        for sentence_index in range(sentence_count):
            scores = scores_by_sentence[sentence_index]
            max_entailment = max(float(row[entailment_index]) for row in scores)
            max_contradiction = max(float(row[contradiction_index]) for row in scores)
            if (
                max_entailment >= NLI_CONFIDENCE_THRESHOLD
                and max_entailment >= max_contradiction
            ):
                categories.append("supported")
            elif (
                max_contradiction >= NLI_CONFIDENCE_THRESHOLD
                and max_contradiction > max_entailment
            ):
                categories.append("contradicted")
            else:
                categories.append("unsupported")
        contradicted_share = categories.count("contradicted") / sentence_count
        backend = "nli_cross_encoder"

    return {
        "accuracy_backend": backend,
        "accuracy_sentence_count": sentence_count,
        "accuracy_supported_share": categories.count("supported") / sentence_count,
        "accuracy_unsupported_share": categories.count("unsupported") / sentence_count,
        "accuracy_contradicted_share": contradicted_share,
    }


def clamp_score(value):
    return max(0.0, min(1.0, float(value)))


def quality_scores_from_metrics(metrics):
    relevance_similarity = metrics["relevance_cosine"]
    relevance_named = metrics["relevance_detected_class_named"]
    relevance = (
        (clamp_score(relevance_similarity) + float(relevance_named)) / 2
        if relevance_similarity is not None else None
    )

    supported = metrics["accuracy_supported_share"]
    if supported is None:
        accuracy = None
    else:
        contradicted = metrics["accuracy_contradicted_share"] or 0.0
        technique_match = metrics["accuracy_technique_id_match_rate"]
        accuracy = clamp_score(supported) * (1 - clamp_score(contradicted))
        if technique_match is not None:
            accuracy *= clamp_score(technique_match)

    mitigation_matches = json.loads(
        metrics["actionability_mitigation_vocabulary_matches"]
    )
    actionability = sum((
        clamp_score(metrics["actionability_required_section_share"]),
        clamp_score(metrics["actionability_action_step_share"] or 0.0),
        clamp_score(metrics["actionability_concrete_action_count"] / 3),
        clamp_score(len(mitigation_matches) / 3),
    )) / 4

    numeric_match = metrics["consistency_numeric_feature_values_match_share"]
    consistency = sum((
        float(metrics["consistency_detected_class_named"]),
        float(metrics["consistency_other_class_diagnosis_absent"]),
        1.0 if numeric_match is None else clamp_score(numeric_match),
    )) / 3

    flesch = metrics["clarity_flesch_reading_ease"]
    word_count = metrics["clarity_word_count"]
    clarity = (
        sum((
            clamp_score(flesch / 100),
            float(metrics["clarity_format_compliant"]),
            1 - clamp_score(metrics["clarity_repeated_trigram_rate"]),
        )) / 3
        if flesch is not None and word_count else None
    )

    return {
        "relevance": relevance,
        "accuracy": accuracy,
        "actionability": actionability,
        "consistency": consistency,
        "clarity": clarity,
    }


class ExplanationScorer:
    def __init__(self, chunks, source_names, source_type_map,
                 known_technique_ids, mitigation_patterns, action_patterns,
                 embedding_model, nli_model):
        self.chunks = chunks
        self.chunk_texts = [chunk["text"] for chunk in chunks]
        self.source_names = source_names
        self.source_type_map = source_type_map
        self.known_technique_ids = known_technique_ids
        self.mitigation_patterns = mitigation_patterns
        self.action_patterns = action_patterns
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        self._lock = Lock()

        self.chunk_embeddings = np.asarray(
            embedding_model.encode(
                self.chunk_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        self.nli_labels = {}
        if nli_model is not None:
            id2label = nli_model.model.config.id2label
            for index, label in id2label.items():
                normalized = str(label).strip().lower()
                if normalized in ("entailment", "contradiction"):
                    self.nli_labels[normalized] = int(index)
            if set(self.nli_labels) != {"entailment", "contradiction"}:
                raise ValueError(f"Unexpected NLI label mapping: {id2label}")

    def evaluate(self, text, attack_type, features, retrieved_chunks=None):
        if not isinstance(text, str):
            raise TypeError("Explanation text must be a string.")
        if not features or not isinstance(features[0], dict):
            raise ValueError("Feature input must be a non-empty list of mappings.")

        label = normalize_label(str(attack_type), "score_explanation")
        retrieved = retrieved_chunks or []
        sentences = split_sentences(text)
        description = feature_description(label, features)

        with self._lock:
            vectors = np.asarray(
                self.embedding_model.encode(
                    [text, description],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
            sentence_vectors = (
                np.asarray(
                    self.embedding_model.encode(
                        sentences,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    ),
                    dtype=np.float32,
                )
                if sentences
                else np.empty(
                    (0, self.chunk_embeddings.shape[1]),
                    dtype=np.float32,
                )
            )

            accuracy = calculate_accuracy(
                sentences,
                sentence_vectors,
                self.chunk_embeddings,
                self.chunk_texts,
                self.nli_model,
                self.nli_labels,
            )

        detected_named = class_is_named(text, label)
        numeric_count, numeric_match_share = numeric_feature_match_share(
            text,
            features,
        )
        row = {
            "text": text,
            "method": "rag" if retrieved else "llm_only",
            "detected_label": label,
            "retrieved_sources": retrieved,
        }
        other_diagnosis_present = other_class_diagnosis_present(text, label)
        return {
            **retrieval_metrics(row, self.source_names, self.source_type_map),
            "relevance_cosine": float(np.dot(vectors[0], vectors[1])),
            "relevance_detected_class_named": detected_named,
            **accuracy,
            **technique_metrics(text, self.known_technique_ids),
            **actionability_metrics(
                text,
                features,
                detected_named,
                self.mitigation_patterns,
                self.action_patterns,
            ),
            "consistency_detected_class_named": detected_named,
            "consistency_other_class_diagnosis_absent": not other_diagnosis_present,
            "consistency_numeric_feature_values_checked": numeric_count,
            "consistency_numeric_feature_values_match_share": numeric_match_share,
            **clarity_metrics(text, sentences),
        }


def create_scorer():
    (
        chunks,
        source_names,
        source_type_map,
        known_technique_ids,
        mitigation_patterns,
        action_patterns,
    ) = load_chunks_and_sources()
    embedding_model, nli_model = load_models()
    return ExplanationScorer(
        chunks,
        source_names,
        source_type_map,
        known_technique_ids,
        mitigation_patterns,
        action_patterns,
        embedding_model,
        nli_model,
    )


@lru_cache(maxsize=1)
def _get_default_scorer():
    return create_scorer()


def score_explanation(text, attack_type, features, retrieved_chunks=None,
                      *, scorer=None, detailed_metrics=None):
    if detailed_metrics is None:
        active_scorer = scorer or _get_default_scorer()
        detailed_metrics = active_scorer.evaluate(
            text,
            attack_type,
            features,
            retrieved_chunks,
        )
    return quality_scores_from_metrics(detailed_metrics)


def evaluate_rows(explanations, chunks, source_names, source_type_map,
                  known_technique_ids, mitigation_patterns, action_patterns,
                  embedding_model, nli_model):
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

    descriptions = [
        feature_description(row["detected_label"], row["features"])
        for row in explanations
    ]
    text_and_description = [
        value
        for row, description in zip(explanations, descriptions)
        for value in (row["text"], description)
    ]
    text_embeddings = embedding_model.encode(
        text_and_description,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)

    sentences_by_row = [split_sentences(row["text"]) for row in explanations]
    all_sentences = [
        sentence
        for sentences in sentences_by_row
        for sentence in sentences
    ]
    sentence_embeddings = embedding_model.encode(
        all_sentences,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ) if all_sentences else np.empty((0, chunk_embeddings.shape[1]), dtype=np.float32)
    sentence_embeddings = np.asarray(sentence_embeddings, dtype=np.float32)

    nli_labels = {}
    if nli_model is not None:
        id2label = nli_model.model.config.id2label
        for index, label in id2label.items():
            normalized = str(label).strip().lower()
            if normalized in ("entailment", "contradiction"):
                nli_labels[normalized] = int(index)
        if set(nli_labels) != {"entailment", "contradiction"}:
            raise ValueError(f"Unexpected NLI label mapping: {id2label}")

    results = []
    sentence_offset = 0
    for index, row in enumerate(explanations):
        text_vector = text_embeddings[index * 2]
        description_vector = text_embeddings[index * 2 + 1]
        similarity = float(np.dot(text_vector, description_vector))
        detected_named = class_is_named(row["text"], row["detected_label"])
        sentences = sentences_by_row[index]
        row_sentence_embeddings = sentence_embeddings[
            sentence_offset:sentence_offset + len(sentences)
        ]
        sentence_offset += len(sentences)

        accuracy = calculate_accuracy(
            sentences,
            row_sentence_embeddings,
            chunk_embeddings,
            chunk_texts,
            nli_model,
            nli_labels,
        )
        numeric_count, numeric_match_share = numeric_feature_match_share(
            row["text"],
            row["features"],
        )
        other_diagnosis_present = other_class_diagnosis_present(
            row["text"],
            row["detected_label"],
        )

        result = {
            "metric_label": "automated proxy metric",
            "sample_id": row["sample_id"],
            "method": row["method"],
            "model": row["model"],
            "detected_label": row["detected_label"],
            **retrieval_metrics(row, source_names, source_type_map),
            "relevance_cosine": similarity,
            "relevance_detected_class_named": detected_named,
            **accuracy,
            **technique_metrics(row["text"], known_technique_ids),
            **actionability_metrics(
                row["text"],
                row["features"],
                detected_named,
                mitigation_patterns,
                action_patterns,
            ),
            "consistency_detected_class_named": detected_named,
            "consistency_other_class_diagnosis_absent": not other_diagnosis_present,
            "consistency_numeric_feature_values_checked": numeric_count,
            "consistency_numeric_feature_values_match_share": numeric_match_share,
            **clarity_metrics(row["text"], sentences),
        }
        quality_scores = score_explanation(
            row["text"],
            row["detected_label"],
            row["features"],
            row["retrieved_sources"],
            detailed_metrics=result,
        )
        result.update({
            f"quality_{name}": quality_scores[name]
            for name in QUALITY_SCORE_NAMES
        })
        results.append(result)
    return results


def resolve_path(path):
    result = Path(path)
    return result if result.is_absolute() else ROOT_DIR / result


def make_output_path(requested_path):
    if requested_path:
        path = resolve_path(requested_path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        return path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAG_DIR / f"evaluation_metrics_{timestamp}.csv"
    suffix = 1
    while path.exists():
        path = RAG_DIR / f"evaluation_metrics_{timestamp}_{suffix:02d}.csv"
        suffix += 1
    return path


def write_results(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute local automated proxy metrics for explanation CSVs."
    )
    parser.add_argument("--comparison", required=True, help="Batch comparison CSV")
    parser.add_argument("--baselines", required=True, help="Non-LLM baselines CSV")
    parser.add_argument("--output", help="New output CSV path (must not already exist)")
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N explanation rows; useful for a smoke test",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main():
    args = parse_args()
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    comparison_path = resolve_path(args.comparison)
    baseline_path = resolve_path(args.baselines)
    comparison_by_sample, baselines_by_sample, detected_by_sample = load_inputs(
        comparison_path,
        baseline_path,
    )

    from rag.get_eval_samples import get_eval_samples

    feature_by_sample = {
        sample["sample_id"]: sample["features"]
        for sample in get_eval_samples()
    }
    explanations = make_explanations(
        comparison_by_sample,
        baselines_by_sample,
        detected_by_sample,
        feature_by_sample,
        args.limit,
    )
    (
        chunks,
        source_names,
        source_type_map,
        known_technique_ids,
        mitigation_patterns,
        action_patterns,
    ) = load_chunks_and_sources()
    embedding_model, nli_model = load_models()
    results = evaluate_rows(
        explanations,
        chunks,
        source_names,
        source_type_map,
        known_technique_ids,
        mitigation_patterns,
        action_patterns,
        embedding_model,
        nli_model,
    )

    output_path = make_output_path(args.output)
    write_results(output_path, results)
    print(f"Accuracy backend: {results[0]['accuracy_backend']}")
    print(
        f"Thresholds: NLI={NLI_CONFIDENCE_THRESHOLD:.2f}, "
        f"embedding fallback cosine={FALLBACK_COSINE_THRESHOLD:.2f}, "
        f"numeric match rel_tol={NUMERIC_REL_TOLERANCE:g}, "
        f"abs_tol={NUMERIC_ABS_TOLERANCE:g}, "
        f"NLI evidence candidates={NLI_CANDIDATE_COUNT}"
    )
    print(f"Output: {output_path}")
    print(f"Explanation rows: {len(results)}")
    for row in results:
        print(
            f"{row['sample_id']} {row['method']} {row['model'] or '-'}: "
            f"relevance={row['relevance_cosine']:.3f}, "
            f"supported={row['accuracy_supported_share']}, "
            f"unsupported={row['accuracy_unsupported_share']}, "
            f"contradicted={row['accuracy_contradicted_share']}"
        )


if __name__ == "__main__":
    main()
