"""
Evaluate cached SFT before/after outputs for content quality.

This script does not load the model. It compares generated Base/LoRA outputs
from outputs/sft_before_after.md against reference outputs in data/sft_test.jsonl.

Outputs:
  outputs/sft_quality_results.md   Summary table and failure examples
  outputs/sft_quality_details.csv  Per-prompt metrics

Usage:
  python src/eval_sft_quality.py
"""

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parent.parent
TEST_PROMPTS_PATH = Path(os.getenv("TEST_PROMPTS_PATH", str(ROOT / "data" / "sft_test.jsonl")))
BEFORE_AFTER_PATH = Path(os.getenv("BEFORE_AFTER_PATH", str(ROOT / "outputs" / "sft_before_after.md")))
QUALITY_RESULTS_PATH = Path(os.getenv("QUALITY_RESULTS_PATH", str(ROOT / "outputs" / "sft_quality_results.md")))
QUALITY_DETAILS_PATH = Path(os.getenv("QUALITY_DETAILS_PATH", str(ROOT / "outputs" / "sft_quality_details.csv")))
QUALITY_REPORT_TITLE = os.getenv("QUALITY_REPORT_TITLE", "Cached Generation Quality Evaluation")
ADAPTER_LABEL = os.getenv("ADAPTER_LABEL", "LoRA")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "being",
    "but", "by", "can", "could", "for", "from", "has", "have", "how", "if",
    "in", "into", "is", "it", "its", "may", "might", "more", "not", "of",
    "on", "or", "rather", "should", "so", "such", "than", "that", "the",
    "their", "then", "there", "these", "this", "those", "to", "when", "where",
    "which", "while", "with", "without", "would",
    "summary", "key", "points", "point", "limitation", "follow", "up",
    "question", "model", "models", "method", "methods", "system", "systems",
}

BOILERPLATE_PATTERNS = [
    "uses a targeted modeling or systems mechanism",
    "connects a concrete technical mechanism with a practical goal",
    "a useful evaluation should include cases",
    "a strong evaluation should include stress cases",
    "validated on realistic held-out examples rather than inferred from the method name alone",
    "the method operates through a setup",
    "the main practical benefit",
    "the most important failure mode appears",
]

GRAMMAR_FLAG_PATTERNS = [
    ("plural_subject_uses", re.compile(r"\b(layers|loops|models|methods|systems|adapters|prompts|networks)\s+uses\b", re.I)),
    ("how_should_ranks", re.compile(r"\bhow should\b[^?]*\branks\b", re.I)),
    ("extra_sentence_after_question", re.compile(r"Follow-up Question\s*:\s*[^?]+\?\s+\w", re.I | re.S)),
]


def content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9_+-]*", text.lower())
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) >= 3]


def counter_f1(candidate: str, reference: str) -> tuple[float, float, float]:
    cand = Counter(content_tokens(candidate))
    ref = Counter(content_tokens(reference))
    if not cand or not ref:
        return 0.0, 0.0, 0.0

    overlap = sum((cand & ref).values())
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def unique_recall(candidate: str, reference: str) -> float:
    cand = set(content_tokens(candidate))
    ref = set(content_tokens(reference))
    if not ref:
        return 0.0
    return len(cand & ref) / len(ref)


def unsupported_rate(candidate: str, input_text: str, reference: str) -> float:
    cand = content_tokens(candidate)
    supported = set(content_tokens(input_text)) | set(content_tokens(reference))
    if not cand:
        return 1.0
    unsupported = [tok for tok in cand if tok not in supported]
    return len(unsupported) / len(cand)


def strict_check_compliance(text: str) -> dict:
    summary = bool(re.search(r"^\s*Summary\s*:", text, re.M | re.I))
    key_points = bool(re.search(r"^\s*Key Points\s*:", text, re.M | re.I))
    limitation = bool(re.search(r"^\s*Limitation\s*:", text, re.M | re.I))

    follow_match = re.search(r"^\s*Follow-up Question\s*:\s*(.*?)$", text, re.S | re.M | re.I)
    follow_text = follow_match.group(1).strip() if follow_match else ""
    follow_up = bool(follow_text) and follow_text.endswith("?")

    kp_match = re.search(r"Key Points\s*:(.*?)(?:Limitation\s*:|$)", text, re.S | re.I)
    bullets = re.findall(r"^\s*[-*]\s+\S", kp_match.group(1), re.M) if kp_match else []
    three_bullets = len(bullets) == 3

    return {
        "summary": summary,
        "key_points": key_points,
        "limitation": limitation,
        "follow_up": follow_up,
        "three_bullets": three_bullets,
        "bullet_count": len(bullets),
        "all_pass": all([summary, key_points, limitation, follow_up, three_bullets]),
    }


def parse_summary(text: str) -> str:
    match = re.search(r"^\s*(?:#+\s*)?(?:\*\*)?Summary(?:\*\*)?\s*:\s*(.*?)(?=\n\s*(?:#+\s*)?(?:\*\*)?Key Points|\Z)", text, re.S | re.M | re.I)
    return match.group(1).strip() if match else ""


def summary_opening(text: str, n_words: int = 6) -> str:
    summary = parse_summary(text)
    tokens = content_tokens(summary)
    return " ".join(tokens[:n_words])


def has_boilerplate(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in BOILERPLATE_PATTERNS)


def grammar_flags(text: str) -> list[str]:
    return [name for name, pattern in GRAMMAR_FLAG_PATTERNS if pattern.search(text)]


def parse_before_after(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text()
    parsed = {}
    blocks = re.split(r"\n##\s+", text)[1:]

    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        record_id = lines[0].strip()
        fenced = re.findall(r"```\n(.*?)```", block, re.S)
        if len(fenced) < 3:
            continue
        parsed[record_id] = {
            "input": fenced[0].strip(),
            "base": fenced[1].strip(),
            "lora": fenced[2].strip(),
        }

    return parsed


def load_references(path: Path) -> dict[str, dict]:
    refs = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            refs[item["id"]] = item
    return refs


def score_output(record: dict, output: str) -> dict:
    precision, recall, f1 = counter_f1(output, record["output"])
    key_recall = unique_recall(output, record["output"])
    unsupported = unsupported_rate(output, record["input"], record["output"])
    compliance = strict_check_compliance(output)
    flags = grammar_flags(output)

    content_alignment = (
        0.60 * f1
        + 0.25 * key_recall
        + 0.15 * (1.0 - unsupported)
    )

    return {
        "format_pass": compliance["all_pass"],
        "followup_valid": compliance["follow_up"],
        "three_bullets": compliance["three_bullets"],
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
        "key_term_recall": key_recall,
        "unsupported_rate": unsupported,
        "content_alignment": content_alignment,
        "word_count": len(output.split()),
        "boilerplate": has_boilerplate(output),
        "grammar_flags": flags,
        "grammar_flag_count": len(flags),
        "summary_opening": summary_opening(output),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def avg_summary(rows: list[dict]) -> dict:
    total = len(rows)
    return {
        "n": total,
        "format": mean([r["format_pass"] for r in rows]),
        "followup": mean([r["followup_valid"] for r in rows]),
        "three_bullets": mean([r["three_bullets"] for r in rows]),
        "content_f1": mean([r["content_f1"] for r in rows]),
        "key_recall": mean([r["key_term_recall"] for r in rows]),
        "unsupported": mean([r["unsupported_rate"] for r in rows]),
        "alignment": mean([r["content_alignment"] for r in rows]),
        "words": mean([r["word_count"] for r in rows]),
        "boilerplate": mean([r["boilerplate"] for r in rows]),
        "grammar_flags": mean([r["grammar_flag_count"] > 0 for r in rows]),
    }


def render_markdown(rows_by_model: dict[str, list[dict]], records_by_id: dict[str, dict]) -> str:
    summaries = {name: avg_summary(rows) for name, rows in rows_by_model.items()}

    lines = [
        f"# {QUALITY_REPORT_TITLE}",
        "",
        f"References: `{TEST_PROMPTS_PATH}`  ",
        f"Cached generations: `{BEFORE_AFTER_PATH}`  ",
        f"Examples: {len(records_by_id)}",
        "",
        "## Summary",
        "",
        "The content metrics compare generated outputs against the held-out reference answer for the same prompt. "
        "`Content Alignment` is a heuristic weighted score: 60% reference-token F1, 25% key-term recall, and 15% supported-token rate.",
        "",
        "| Model | Format Compliance | Content Alignment | Ref Token F1 | Key-Term Recall | Unsupported Terms | Follow-up Valid | Formulaic Rate | Grammar Flags | Avg Words |",
        "|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|",
    ]

    display_names = {"base": "base", "lora": ADAPTER_LABEL}
    for model_name in ["base", "lora"]:
        s = summaries[model_name]
        lines.append(
            f"| {display_names[model_name]} "
            f"| {pct(s['format'])} "
            f"| {pct(s['alignment'])} "
            f"| {pct(s['content_f1'])} "
            f"| {pct(s['key_recall'])} "
            f"| {pct(s['unsupported'])} "
            f"| {pct(s['followup'])} "
            f"| {pct(s['boilerplate'])} "
            f"| {pct(s['grammar_flags'])} "
            f"| {s['words']:.1f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `Format Compliance` is the strict template check: exact section headers, exactly three bullets, and a follow-up that ends with `?`.",
        "- `Content Alignment` is not human judgment; it checks overlap with the held-out reference answer and whether generated terms are supported by the prompt/reference.",
        "- `Formulaic Rate` is a warning signal for repeated learned phrasing. High format compliance with high formulaic rate usually means the next improvement should be DPO or style-diverse SFT data.",
        "",
        f"## {ADAPTER_LABEL} Failure / Risk Examples",
        "",
    ]

    lora_rows = rows_by_model["lora"]
    failures = [row for row in lora_rows if not row["format_pass"] or row["grammar_flags"]]
    if failures:
        lines += ["| Prompt | Issue | Follow-up / Opening |", "|--------|-------|---------------------|"]
        for row in failures[:12]:
            issues = []
            if not row["format_pass"]:
                issues.append("format")
            issues.extend(row["grammar_flags"])
            display = row["followup_text"] or row["summary_opening"]
            lines.append(f"| {row['id']} | {', '.join(issues)} | {display} |")
    else:
        lines.append(f"No {ADAPTER_LABEL} format or grammar flags found by the heuristic checks.")

    lines += [
        "",
        f"## Lowest {ADAPTER_LABEL} Content-Alignment Examples",
        "",
        "| Prompt | Topic | Difficulty | Content Alignment | Ref Token F1 | Key-Term Recall |",
        "|--------|-------|------------|:---:|:---:|:---:|",
    ]

    for row in sorted(lora_rows, key=lambda r: r["content_alignment"])[:10]:
        record = records_by_id[row["id"]]
        lines.append(
            f"| {row['id']} | {record.get('topic', '')} | {record.get('difficulty', '')} "
            f"| {pct(row['content_alignment'])} | {pct(row['content_f1'])} | {pct(row['key_term_recall'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def extract_followup(text: str) -> str:
    match = re.search(r"^\s*Follow-up Question\s*:\s*(.*?)$", text, re.S | re.M | re.I)
    return match.group(1).strip().replace("\n", " ") if match else ""


def main() -> None:
    references = load_references(TEST_PROMPTS_PATH)
    generated = parse_before_after(BEFORE_AFTER_PATH)

    missing = sorted(set(references) - set(generated))
    extra = sorted(set(generated) - set(references))
    if missing or extra:
        raise SystemExit(
            f"Reference/generated ID mismatch. Missing={missing[:5]} Extra={extra[:5]}"
        )

    rows_by_model = {"base": [], "lora": []}
    detail_rows = []

    for record_id, record in references.items():
        gen = generated[record_id]
        for model_name in ["base", "lora"]:
            metrics = score_output(record, gen[model_name])
            row = {
                "id": record_id,
                "model": model_name,
                "topic": record.get("topic", ""),
                "difficulty": record.get("difficulty", ""),
                "followup_text": extract_followup(gen[model_name]),
                **metrics,
            }
            rows_by_model[model_name].append(row)
            detail_rows.append(row)

    QUALITY_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_RESULTS_PATH.write_text(render_markdown(rows_by_model, references))

    fieldnames = [
        "id", "model", "topic", "difficulty", "format_pass", "followup_valid",
        "three_bullets", "content_precision", "content_recall", "content_f1",
        "key_term_recall", "unsupported_rate", "content_alignment", "word_count",
        "boilerplate", "grammar_flag_count", "grammar_flags", "summary_opening",
    ]
    with QUALITY_DETAILS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detail_rows:
            csv_row = {key: row.get(key, "") for key in fieldnames}
            csv_row["grammar_flags"] = ";".join(row["grammar_flags"])
            writer.writerow(csv_row)

    print(f"Quality report written to: {QUALITY_RESULTS_PATH}")
    print(f"Quality details written to: {QUALITY_DETAILS_PATH}")


if __name__ == "__main__":
    main()
