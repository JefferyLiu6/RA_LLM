"""
Build a DPO preference-pair dataset from train/val SFT records.

The default build keeps the held-out test split untouched. Each pair keeps the
original prompt, uses the validated reference output as the chosen response,
and creates hard near-miss rejected responses from that same reference answer.
The rejected answers mostly preserve topic and wording, but introduce one
targeted flaw such as an extra bullet, unsupported claim, or invalid follow-up.

Output schema:
  {
    "id": "dpo_0001",
    "prompt": "...",
    "chosen": "...",
    "rejected": "...",
    "source_id": "...",
    "source_split": "train",
    "topic": "...",
    "difficulty": "...",
    "pair_source": "reference_output_vs_hard_near_miss_negative",
    "issue_tags": ["extra_bullet", ...],
    "input_sha256": "..."
  }

Usage:
  python src/build_dpo_pairs.py
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parent.parent
DEFAULT_INPUTS = [ROOT / "data" / "sft_train.jsonl", ROOT / "data" / "sft_val.jsonl"]
DEFAULT_OUTPUT = ROOT / "data" / "dpo_pairs.jsonl"
DEFAULT_REPORT = ROOT / "outputs" / "dpo_pair_report.md"
DEFAULT_MAX_PAIRS = 300
DEFAULT_PAIRS_PER_RECORD = 3

UNSUPPORTED_CLAIMS = [
    "The same team also used an active-learning curriculum that is not described in the prompt.",
    "The result depends on a larger proprietary reward model that the prompt never mentions.",
    "The method also requires online user labels from deployment traffic.",
    "The benchmark assumes chain-of-thought traces are available for every example.",
]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_clause(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.rstrip(" .;:")


def sentence_case(text: str) -> str:
    text = clean_clause(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def concept_from_input(text: str) -> str:
    markers = [
        " is usually discussed",
        " appears in production systems",
        " is a useful case study",
        " matters in research pipelines",
        " creates a trade-off",
        " should be stress-tested",
        " is often introduced",
        " can be evaluated",
    ]
    for marker in markers:
        if marker in text:
            return clean_clause(text.split(marker, 1)[0])
    return clean_clause(" ".join(text.split()[:4]))


def parse_reference_sections(output: str) -> dict:
    def grab(name: str, stop: str | None) -> str:
        if stop:
            pattern = rf"{name}\s*:\s*(.*?)(?=\n{stop}\s*:|\Z)"
        else:
            pattern = rf"{name}\s*:\s*(.*?)$"
        match = re.search(pattern, output, re.S | re.I)
        return match.group(1).strip() if match else ""

    key_points = grab("Key Points", "Limitation")
    bullets = re.findall(r"^\s*[-*]\s+(.+)", key_points, re.M)
    return {
        "summary": grab("Summary", "Key Points"),
        "bullets": bullets,
        "limitation": grab("Limitation", "Follow-up Question"),
        "followup": grab("Follow-up Question", None),
    }


def ensure_sentence(text: str) -> str:
    text = clean_clause(text)
    if not text:
        return text
    if text.endswith(("?", "!", ".")):
        return text
    return f"{text}."


def render_response(summary: str, bullets: list[str], limitation: str, followup: str) -> str:
    rendered_bullets = [ensure_sentence(bullet) for bullet in bullets]
    return "\n".join([
        "Summary:",
        ensure_sentence(summary),
        "",
        "Key Points:",
        *[f"- {bullet}" for bullet in rendered_bullets],
        "",
        "Limitation:",
        ensure_sentence(limitation),
        "",
        "Follow-up Question:",
        followup.strip(),
    ])


def sections_or_rewrite(record: dict, fallback_index: int) -> tuple[str, dict]:
    chosen = record["output"].strip()
    sections = parse_reference_sections(chosen)
    if sections["summary"] and len(sections["bullets"]) >= 3 and sections["limitation"]:
        return chosen, sections

    chosen = build_chosen(record, fallback_index)
    sections = parse_reference_sections(chosen)
    return chosen, sections


def normalize_mechanism(text: str) -> str:
    text = clean_clause(text)
    prefixes = [
        "The method should be explained in terms of its mechanism:",
        "The method operates through a setup where",
        "Its core mechanism is that",
        "The core mechanism is that",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return clean_clause(text[len(prefix):])
    return text


def normalize_benefit(text: str) -> str:
    text = clean_clause(text)
    prefixes = [
        "Its expected upside is that teams can",
        "The main practical benefit is that it can",
        "The main practical benefit is that",
        "It is most valuable in workflows that need to",
        "It offers value in workflows that need to",
        "This helps teams to",
        "This helps teams",
        "Teams can",
        "It can",
        "Can",
        "To",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return clean_clause(text[len(prefix):])
    return text


def normalize_risk(text: str) -> str:
    text = clean_clause(text)
    prefixes = [
        "A strong evaluation should include stress cases where",
        "A useful evaluation should include cases where",
        "The most important failure mode appears when",
        "Evaluation risk:",
        "Risk to test:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return clean_clause(text[len(prefix):])
    return text


def strip_leading_that(text: str) -> str:
    text = clean_clause(text)
    if text.lower().startswith("that "):
        return clean_clause(text[5:])
    return text


def parse_input_facts(text: str, output: str) -> dict:
    concept = concept_from_input(text)
    compact = re.sub(r"\s+", " ", text.strip())

    patterns = [
        (
            r"^(?P<concept>.+?) is usually discussed.*?In a realistic system, (?P<mechanism>.+?), which can (?P<benefit>.+?)\. The hard part.*? when (?P<risk>.+?)\.$",
            ("concept", "mechanism", "benefit", "risk"),
        ),
        (
            r"^(?P<concept>.+?) appears in production systems when engineers want to (?P<benefit>.+?)\. The implementation usually relies on the fact that (?P<mechanism>.+?); the operational risk is that (?P<risk>.+?)\.$",
            ("concept", "benefit", "mechanism", "risk"),
        ),
        (
            r"^(?P<concept>.+?) is a useful case study.*? The key technical idea is that (?P<mechanism>.+?)\. This matters because it can (?P<benefit>.+?), but.*? if (?P<risk>.+?)\.$",
            ("concept", "mechanism", "benefit", "risk"),
        ),
        (
            r"^(?P<concept>.+?) matters in research pipelines because it offers a way to (?P<benefit>.+?)\. The underlying mechanism is that (?P<mechanism>.+?)\. Any benchmark should include examples where (?P<risk>.+?)\.$",
            ("concept", "benefit", "mechanism", "risk"),
        ),
        (
            r"^(?P<concept>.+?) creates a trade-off.*? It can (?P<benefit>.+?) because (?P<mechanism>.+?), but.*? when (?P<risk>.+?)\.$",
            ("concept", "benefit", "mechanism", "risk"),
        ),
        (
            r"^(?P<concept>.+?) should be stress-tested.*? The method can (?P<benefit>.+?) by relying on the mechanism that (?P<mechanism>.+?), but failures are likely when (?P<risk>.+?)\.$",
            ("concept", "benefit", "mechanism", "risk"),
        ),
        (
            r"^(?P<concept>.+?) is often introduced.*?: (?P<mechanism>.+?)\. Its value is that it can (?P<benefit>.+?), though.*? where (?P<risk>.+?)\.$",
            ("concept", "mechanism", "benefit", "risk"),
        ),
        (
            r"^(?P<concept>.+?) can be evaluated by asking whether the mechanism, where (?P<mechanism>.+?), actually supports the intended benefit: (?P<benefit>.+?)\. A negative result would often show up when (?P<risk>.+?)\.$",
            ("concept", "mechanism", "benefit", "risk"),
        ),
    ]

    facts = {"concept": concept, "mechanism": "", "benefit": "", "risk": ""}
    for pattern, _ in patterns:
        match = re.match(pattern, compact)
        if match:
            facts.update({key: clean_clause(value) for key, value in match.groupdict().items()})
            break

    sections = parse_reference_sections(output)
    if not facts["mechanism"] and sections["bullets"]:
        facts["mechanism"] = normalize_mechanism(sections["bullets"][0].split(":", 1)[-1])
    if not facts["benefit"] and len(sections["bullets"]) > 1:
        facts["benefit"] = normalize_benefit(sections["bullets"][1].split(":", 1)[-1])
    if not facts["risk"] and sections["limitation"]:
        facts["risk"] = normalize_risk(sections["limitation"].split(";")[0])

    facts["concept"] = strip_leading_that(clean_clause(facts["concept"]))
    facts["mechanism"] = strip_leading_that(normalize_mechanism(facts["mechanism"])) or "the core mechanism is applied to the prompt"
    facts["benefit"] = strip_leading_that(normalize_benefit(facts["benefit"])) or "produce a more reliable structured note"
    facts["risk"] = strip_leading_that(normalize_risk(facts["risk"])) or "the method can fail on realistic edge cases"
    facts["followup"] = sections["followup"] if sections["followup"].endswith("?") else ""
    return facts


def generated_followup(facts: dict) -> str:
    concept = facts["concept"].lower()
    risk = facts["risk"].rstrip(".")
    return f"What evaluation would show whether {concept} still works when {risk}?"


def build_chosen(record: dict, index: int) -> str:
    facts = parse_input_facts(record["input"], record["output"])
    concept = facts["concept"]
    mechanism = facts["mechanism"]
    benefit = facts["benefit"]
    risk = facts["risk"]
    followup = facts["followup"] or generated_followup(facts)

    style = index % 4
    if style == 0:
        summary = (
            f"In practice, {concept} helps teams to {benefit} by relying on the fact that {mechanism}. "
            f"The main risk is that {risk}."
        )
    elif style == 1:
        summary = (
            f"{concept} is useful for teams trying to {benefit}. "
            f"It works through the fact that {mechanism}, and it should be tested where {risk}."
        )
    elif style == 2:
        summary = (
            f"The value of {concept} is that it helps teams to {benefit}. "
            f"Its behavior depends on the fact that {mechanism}, especially when {risk}."
        )
    else:
        summary = (
            f"For research notes, {concept} should be explained through the fact that {mechanism}, "
            f"the benefit of {benefit}, and the risk that {risk}."
        )

    bullets = [
        f"Mechanism: {sentence_case(mechanism)}.",
        f"Use case: this helps teams {benefit}.",
        f"Evaluation risk: {sentence_case(risk)}.",
    ]
    limitation = (
        f"{sentence_case(risk)}, so the evaluation should include examples that expose "
        "that failure mode rather than only average-case prompts."
    )

    return "\n".join([
        "Summary:",
        summary,
        "",
        "Key Points:",
        f"- {bullets[0]}",
        f"- {bullets[1]}",
        f"- {bullets[2]}",
        "",
        "Limitation:",
        limitation,
        "",
        "Follow-up Question:",
        followup,
    ])


def build_legacy_rejected(record: dict, index: int) -> tuple[str, list[str], str]:
    facts = parse_input_facts(record["input"], record["output"])
    concept = facts["concept"]
    concept_l = concept.lower()
    mechanism = facts["mechanism"]
    benefit = facts["benefit"]
    risk = facts["risk"]

    tags = ["formulaic_summary", "boilerplate_limitation", "repetitive_bullets"]
    followup_templates = [
        (f"How should {concept_l} ranks examples before deployment?", "grammar_followup"),
        (f"How does {concept_l} affect performance for small models?", "generic_followup"),
        (
            f"How much {concept_l} is too much? Teams often decide this qualitatively rather than with a hard number.",
            "extra_sentence_after_question",
        ),
    ]
    followup, followup_tag = followup_templates[index % len(followup_templates)]
    tags.append(followup_tag)

    rejected = "\n".join([
        "Summary:",
        (
            f"{concept} uses a targeted modeling or systems mechanism to {benefit}, "
            f"while requiring care because {risk}."
        ),
        "",
        "Key Points:",
        f"- The method operates through a setup where {mechanism}.",
        f"- The main practical benefit is that it can {benefit}.",
        f"- The most important failure mode appears when {risk}.",
        "",
        "Limitation:",
        (
            f"{sentence_case(risk)}, which means benchmark results can look stronger "
            "than real deployment behavior."
        ),
        "",
        "Follow-up Question:",
        followup,
    ])

    return rejected, tags, "synthetic_formulaic_negative"


def build_hard_rejections(record: dict, chosen: str, index: int) -> list[tuple[str, list[str], str]]:
    facts = parse_input_facts(record["input"], chosen)
    sections = parse_reference_sections(chosen)
    summary = sections["summary"] or parse_reference_sections(build_chosen(record, index))["summary"]
    bullets = sections["bullets"][:3]
    limitation = sections["limitation"]
    followup = sections["followup"] if sections["followup"].endswith("?") else generated_followup(facts)

    if len(bullets) < 3:
        bullets = [
            f"Mechanism: {sentence_case(facts['mechanism'])}.",
            f"Use case: this helps teams {facts['benefit']}.",
            f"Evaluation risk: {sentence_case(facts['risk'])}.",
        ]
    if not limitation:
        limitation = (
            f"{sentence_case(facts['risk'])}, so the evaluation should include examples "
            "that expose that failure mode."
        )

    unsupported = UNSUPPORTED_CLAIMS[index % len(UNSUPPORTED_CLAIMS)]
    generic_followup = f"How does {facts['concept'].lower()} affect performance for small models?"

    extra_bullet = render_response(
        summary,
        [*bullets[:3], unsupported],
        limitation,
        followup,
    )

    extra_followup_sentence = render_response(
        summary,
        bullets[:3],
        limitation,
        (
            f"{followup} Teams often decide this qualitatively rather than with a "
            "held-out evaluation."
        ),
    )

    unsupported_content = render_response(
        summary,
        [bullets[0], unsupported, bullets[2]],
        limitation,
        followup,
    )

    generic_limitation = render_response(
        summary,
        bullets[:3],
        "More testing is needed before using this method in production.",
        generic_followup,
    )

    formulaic = "\n".join([
        "Summary:",
        (
            f"{facts['concept']} uses a targeted modeling or systems mechanism to {facts['benefit']}, "
            f"while requiring care because {facts['risk']}."
        ),
        "",
        "Key Points:",
        f"- The method operates through a setup where {facts['mechanism']}.",
        f"- The main practical benefit is that it can {facts['benefit']}.",
        f"- The most important failure mode appears when {facts['risk']}.",
        "",
        "Limitation:",
        (
            f"{sentence_case(facts['risk'])}, which means benchmark results can look stronger "
            "than real deployment behavior."
        ),
        "",
        "Follow-up Question:",
        f"How should {facts['concept'].lower()} ranks examples before deployment?",
    ])

    return [
        (
            extra_bullet,
            ["extra_bullet", "unsupported_claim", "near_miss_reference_negative"],
            "hard_negative_extra_bullet",
        ),
        (
            extra_followup_sentence,
            ["extra_sentence_after_question", "near_miss_reference_negative"],
            "hard_negative_extra_followup_sentence",
        ),
        (
            unsupported_content,
            ["unsupported_claim", "content_drift", "format_pass_negative"],
            "hard_negative_unsupported_content",
        ),
        (
            generic_limitation,
            ["generic_limitation", "generic_followup", "low_specificity", "format_pass_negative"],
            "hard_negative_generic_quality",
        ),
        (
            formulaic,
            ["formulaic_summary", "boilerplate_limitation", "grammar_followup"],
            "synthetic_formulaic_negative",
        ),
    ]


def format_pass(text: str) -> bool:
    has_summary = bool(re.search(r"^\s*Summary\s*:", text, re.M | re.I))
    has_key_points = bool(re.search(r"^\s*Key Points\s*:", text, re.M | re.I))
    has_limitation = bool(re.search(r"^\s*Limitation\s*:", text, re.M | re.I))
    follow_match = re.search(r"^\s*Follow-up Question\s*:\s*(.*?)$", text, re.S | re.M | re.I)
    follow_text = follow_match.group(1).strip() if follow_match else ""
    has_followup = bool(follow_text) and follow_text.endswith("?")
    kp_match = re.search(r"Key Points\s*:(.*?)(?:Limitation\s*:|$)", text, re.S | re.I)
    bullet_count = len(re.findall(r"^\s*[-*]\s+\S", kp_match.group(1), re.M)) if kp_match else 0
    return all([has_summary, has_key_points, has_limitation, has_followup, bullet_count == 3])


def balanced_select(records: list[dict], max_pairs: int) -> list[dict]:
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["id"]):
        groups[(record.get("topic", ""), record.get("difficulty", ""))].append(record)

    selected = []
    keys = sorted(groups)
    while len(selected) < max_pairs and any(groups.values()):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                if len(selected) >= max_pairs:
                    break
    return selected


def build_pairs(
    records: list[dict],
    max_pairs: int,
    chosen_mode: str,
    negative_mode: str,
    pairs_per_record: int,
) -> list[dict]:
    records_needed = max(1, math.ceil(max_pairs / max(1, pairs_per_record)))
    selected = balanced_select(records, min(records_needed, len(records)))
    pairs = []
    for record_idx, record in enumerate(selected, 1):
        if chosen_mode == "rewrite":
            chosen = build_chosen(record, record_idx - 1)
            chosen_source = "reference_facts_rewritten"
            pair_source_prefix = "reference_rewrite"
        else:
            chosen, _ = sections_or_rewrite(record, record_idx - 1)
            chosen_source = "reference_output"
            pair_source_prefix = "reference_output"

        if negative_mode == "legacy":
            negative_variants = [build_legacy_rejected(record, record_idx - 1)]
        else:
            negative_variants = build_hard_rejections(record, chosen, record_idx - 1)

        for rejected, tags, rejected_source in negative_variants[:pairs_per_record]:
            pair_idx = len(pairs) + 1
            pairs.append({
                "id": f"dpo_{pair_idx:04d}",
                "prompt": record["input"],
                "chosen": chosen,
                "rejected": rejected,
                "source_id": record["id"],
                "source_split": record.get("split", ""),
                "topic": record.get("topic", ""),
                "difficulty": record.get("difficulty", ""),
                "pair_source": f"{pair_source_prefix}_vs_{rejected_source}",
                "chosen_source": chosen_source,
                "rejected_source": rejected_source,
                "issue_tags": tags,
                "input_sha256": record.get("input_sha256", ""),
            })
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    return pairs


def render_report(pairs: list[dict], input_paths: list[Path], output_path: Path) -> str:
    split_counts = Counter(pair["source_split"] for pair in pairs)
    topic_counts = Counter(pair["topic"] for pair in pairs)
    difficulty_counts = Counter(pair["difficulty"] for pair in pairs)
    rejected_source_counts = Counter(pair["rejected_source"] for pair in pairs)
    tag_counts = Counter(tag for pair in pairs for tag in pair["issue_tags"])
    chosen_format_pass = sum(format_pass(pair["chosen"]) for pair in pairs)
    rejected_format_pass = sum(format_pass(pair["rejected"]) for pair in pairs)

    def rows(counter: Counter) -> list[str]:
        return [f"| {key} | {value} |" for key, value in sorted(counter.items())]

    lines = [
        "# DPO Pair Build Report",
        "",
        f"Output: `{output_path}`",
        f"Pairs: {len(pairs)}",
        "Input files:",
    ]
    lines += [f"- `{path}`" for path in input_paths]
    lines += [
        "",
        "## Split Counts",
        "",
        "| Split | Count |",
        "|-------|------:|",
        *rows(split_counts),
        "",
        "## Topic Counts",
        "",
        "| Topic | Count |",
        "|-------|------:|",
        *rows(topic_counts),
        "",
        "## Difficulty Counts",
        "",
        "| Difficulty | Count |",
        "|------------|------:|",
        *rows(difficulty_counts),
        "",
        "## Rejected-Issue Tags",
        "",
        "| Tag | Count |",
        "|-----|------:|",
        *rows(tag_counts),
        "",
        "## Rejected Source Counts",
        "",
        "| Source | Count |",
        "|--------|------:|",
        *rows(rejected_source_counts),
        "",
        "## Validation",
        "",
        f"- Unique pair ids: {len({pair['id'] for pair in pairs})}/{len(pairs)}",
        f"- Test-split pairs: {sum(pair['source_split'] == 'test' for pair in pairs)}",
        f"- Chosen strict-format pass: {chosen_format_pass}/{len(pairs)}",
        f"- Rejected strict-format pass: {rejected_format_pass}/{len(pairs)}",
        "",
        "## Notes",
        "",
        "- The held-out test split is not used by default.",
        "- `chosen` responses use the validated train/val reference output by default.",
        "- Default `rejected` responses are hard near-misses made from the same reference answer.",
        "- The hard negatives target observed DPO failures: extra bullets, unsupported claims, and extra text after the follow-up question.",
        "- Some rejected responses intentionally pass strict format so DPO also learns content quality rather than only schema compliance.",
        "- Use `--negative-mode legacy` to reproduce the older formulaic-negative build.",
        "- Pass `--chosen-mode rewrite` only for experimental local data generation; review those pairs carefully.",
        "- Review a sample manually before using these pairs for a final DPO claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs.")
    parser.add_argument("--input", action="append", type=Path, default=[], help="Input JSONL split. Can be repeated.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--chosen-mode", choices=["reference", "rewrite"], default="reference")
    parser.add_argument("--negative-mode", choices=["hard", "legacy"], default="hard")
    parser.add_argument("--pairs-per-record", type=int, default=DEFAULT_PAIRS_PER_RECORD)
    parser.add_argument("--allow-test", action="store_true", help="Allow records marked split=test.")
    args = parser.parse_args()

    if args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1")
    if args.pairs_per_record < 1:
        raise SystemExit("--pairs-per-record must be >= 1")

    input_paths = args.input or DEFAULT_INPUTS
    records = []
    for path in input_paths:
        records.extend(read_jsonl(path))

    test_records = [record["id"] for record in records if record.get("split") == "test"]
    if test_records and not args.allow_test:
        raise SystemExit(
            f"Refusing to build DPO pairs from test records without --allow-test. "
            f"Example IDs: {test_records[:5]}"
        )

    pairs = build_pairs(
        records,
        args.max_pairs,
        args.chosen_mode,
        args.negative_mode,
        args.pairs_per_record,
    )
    write_jsonl(args.output, pairs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(pairs, input_paths, args.output))

    print(f"DPO pairs written to: {args.output}")
    print(f"DPO report written to: {args.report}")
    print(f"Pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
