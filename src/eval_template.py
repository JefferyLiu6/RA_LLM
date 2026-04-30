"""
Evaluate template-adherence of base model vs LoRA adapter on held-out prompts.

Checks each output for:
  - Summary section present
  - Key Points section present with exactly 3 bullet points
  - Limitation section present
  - Follow-up Question section present and ending with '?'

Prints a per-prompt table and writes outputs/eval_results.md.

Usage:
  python src/eval_template.py

  # Re-generate model outputs even if before_after.md exists:
  python src/eval_template.py --regenerate
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import SYSTEM_PROMPT

ROOT = Path(__file__).parent.parent
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", str(ROOT / "outputs" / "lora_adapter")))
TEST_PROMPTS_PATH = Path(os.getenv("TEST_PROMPTS_PATH", str(ROOT / "data" / "test_prompts.jsonl")))
BEFORE_AFTER_PATH = Path(os.getenv("BEFORE_AFTER_PATH", str(ROOT / "outputs" / "before_after.md")))
EVAL_RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", str(ROOT / "outputs" / "eval_results.md")))
OUTPUT_DIR = ROOT / "outputs"

GEN_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))
GEN_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
GEN_DO_SAMPLE = os.getenv("DO_SAMPLE", "0") == "1"
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0"))
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
ADAPTER_LABEL = os.getenv("ADAPTER_LABEL", "LoRA")
EVAL_IDS = {
    item.strip()
    for item in os.getenv("EVAL_IDS", "").split(",")
    if item.strip()
}


# ---------------------------------------------------------------------------
# Template compliance checks
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    ("summary", re.compile(r"^\s*Summary\s*:", re.MULTILINE | re.IGNORECASE)),
    ("key_points", re.compile(r"^\s*Key Points\s*:", re.MULTILINE | re.IGNORECASE)),
    ("limitation", re.compile(r"^\s*Limitation\s*:", re.MULTILINE | re.IGNORECASE)),
]
FOLLOW_UP_RE = re.compile(r"^\s*Follow-up Question\s*:\s*(.*?)$", re.DOTALL | re.MULTILINE | re.IGNORECASE)

BULLET_RE = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)


def check_compliance(text: str) -> dict:
    """Return a dict of individual check results and an overall pass/fail."""
    results = {}

    for name, pattern in REQUIRED_SECTIONS:
        results[name] = bool(pattern.search(text))

    follow_up_match = FOLLOW_UP_RE.search(text)
    follow_up_text = follow_up_match.group(1).strip() if follow_up_match else ""
    results["follow_up"] = bool(follow_up_text) and follow_up_text.endswith("?")

    # Count bullet points that appear after 'Key Points:'
    kp_match = re.search(r"Key Points\s*:(.*?)(?:Limitation\s*:|$)", text, re.DOTALL | re.IGNORECASE)
    if kp_match:
        bullets = BULLET_RE.findall(kp_match.group(1))
        results["three_bullets"] = len(bullets) == 3
        results["bullet_count"] = len(bullets)
    else:
        results["three_bullets"] = False
        results["bullet_count"] = 0

    results["all_pass"] = all(
        results[k] for k in ("summary", "key_points", "limitation", "follow_up", "three_bullets")
    )
    return results


def compliance_score(outputs: list[str]) -> tuple[int, list[dict]]:
    """Return (number of fully-compliant outputs, list of per-output check dicts)."""
    checks = [check_compliance(o) for o in outputs]
    passed = sum(1 for c in checks if c["all_pass"])
    return passed, checks


# ---------------------------------------------------------------------------
# Model loading / generation (reused if outputs not yet available)
# ---------------------------------------------------------------------------

def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_base_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model.eval()
    return model, tokenizer


def build_prompt(user_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt_text: str, device: str) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "do_sample": GEN_DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if GEN_DO_SAMPLE:
        generation_kwargs["temperature"] = GEN_TEMPERATURE

    with torch.no_grad():
        out = model.generate(
            **inputs,
            **generation_kwargs,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_eval_prompts() -> list[dict]:
    with open(TEST_PROMPTS_PATH) as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    if EVAL_IDS:
        prompts = [prompt for prompt in prompts if prompt.get("id") in EVAL_IDS]
        missing = sorted(EVAL_IDS - {prompt.get("id") for prompt in prompts})
        if missing:
            raise ValueError(f"EVAL_IDS not found in {TEST_PROMPTS_PATH}: {missing}")

    if EVAL_LIMIT > 0:
        prompts = prompts[:EVAL_LIMIT]

    return prompts


def run_inference_for_eval(model_id: str, prompts: list[dict]) -> tuple[list[str], list[str]]:
    """Generate base and lora outputs for all test prompts."""
    device = detect_device()
    print(f"Device: {device}")

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}\n"
            "Run `python src/train_lora.py` first."
        )

    print(f"Loading base model {model_id}...")
    base_model, tokenizer = load_base_model(model_id, device)

    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    lora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    lora_model.eval()

    base_outputs, lora_outputs = [], []

    for i, ex in enumerate(prompts, 1):
        user_text = ex["input"]
        prompt_str = build_prompt(user_text, tokenizer)
        print(f"  [{i}/{len(prompts)}] {ex.get('id', '')}...")

        lora_model.disable_adapter_layers()
        base_out = generate(lora_model, tokenizer, prompt_str, device)
        lora_model.enable_adapter_layers()
        lora_out = generate(lora_model, tokenizer, prompt_str, device)

        base_outputs.append(base_out)
        lora_outputs.append(lora_out)

    return base_outputs, lora_outputs


def parse_outputs_from_report(report_path: Path) -> tuple[list[str], list[str]]:
    """
    Parse before_after.md to extract base and lora outputs without rerunning inference.
    Expects fenced code blocks alternating base/lora per prompt.
    """
    text = report_path.read_text()
    blocks = re.split(r"\n##\s+[^\n]+", text)[1:]  # skip preamble

    if not blocks:
        blocks = re.split(r"\n## test_\d+", text)[1:]  # legacy report format

    base_outputs, lora_outputs = [], []
    for block in blocks:
        fenced = re.findall(r"```\n(.*?)```", block, re.DOTALL)
        if len(fenced) >= 3:
            base_outputs.append(fenced[1].strip())
            lora_outputs.append(fenced[2].strip())
        elif len(fenced) >= 2:
            base_outputs.append(fenced[0].strip())
            lora_outputs.append(fenced[1].strip())
        else:
            base_outputs.append("")
            lora_outputs.append("")

    return base_outputs, lora_outputs


def render_before_after_report(
    prompt_ids: list[str],
    prompts: list[dict],
    base_outputs: list[str],
    lora_outputs: list[str],
    model_id: str,
) -> str:
    sections = [
        "# Before / After Evaluation Outputs",
        "",
        f"Base model: `{model_id}`  ",
        f"Adapter: `{ADAPTER_DIR}`  ",
        f"Prompts: `{TEST_PROMPTS_PATH}`  ",
        f"Count: {len(prompts)}",
        "",
    ]

    for pid, prompt, base_out, lora_out in zip(prompt_ids, prompts, base_outputs, lora_outputs):
        sections.extend([
            f"## {pid}",
            "",
            "### Input",
            "",
            "```",
            prompt["input"].strip(),
            "```",
            "",
            "### Base",
            "",
            "```",
            base_out.strip(),
            "```",
            "",
            "### LoRA",
            "",
            "```",
            lora_out.strip(),
            "```",
            "",
        ])

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

SECTION_NAMES = {
    "summary": "Summary",
    "key_points": "Key Points",
    "three_bullets": "3 Bullets",
    "limitation": "Limitation",
    "follow_up": "Follow-up Q",
}


def symbol(flag: bool) -> str:
    return "Y" if flag else "N"


def render_table(
    prompt_ids: list[str],
    base_checks: list[dict],
    lora_checks: list[dict],
) -> str:
    col_keys = ["summary", "key_points", "three_bullets", "limitation", "follow_up", "all_pass"]
    col_labels = ["Summary", "Key Points", "3 Bullets", "Limitation", "Follow-up Q", "PASS"]

    header = "| Prompt | " + " | ".join(f"Base {l}" for l in col_labels) + " | " + " | ".join(f"{ADAPTER_LABEL} {l}" for l in col_labels) + " |"
    sep = "|--------|" + "|".join(["-------"] * len(col_keys) * 2) + "|"

    rows = [header, sep]
    for pid, bc, lc in zip(prompt_ids, base_checks, lora_checks):
        base_cells = " | ".join(symbol(bc.get(k, False)) for k in col_keys)
        lora_cells = " | ".join(symbol(lc.get(k, False)) for k in col_keys)
        rows.append(f"| {pid} | {base_cells} | {lora_cells} |")

    return "\n".join(rows)


def render_summary(base_checks: list[dict], lora_checks: list[dict], n: int) -> str:
    base_pass = sum(1 for c in base_checks if c["all_pass"])
    lora_pass = sum(1 for c in lora_checks if c["all_pass"])

    lines = [
        "## Summary",
        "",
        f"| Model | Fully Compliant | Score |",
        f"|-------|----------------|-------|",
        f"| Base  | {base_pass}/{n}            | {base_pass/n*100:.0f}%   |",
        f"| {ADAPTER_LABEL}  | {lora_pass}/{n}            | {lora_pass/n*100:.0f}%   |",
        "",
    ]

    # Per-section breakdown
    lines += ["## Per-Section Compliance", ""]
    lines += [f"| Section | Base | {ADAPTER_LABEL} |", "|---------|------|------|"]
    for key, label in SECTION_NAMES.items():
        base_sect = sum(1 for c in base_checks if c.get(key, False))
        lora_sect = sum(1 for c in lora_checks if c.get(key, False))
        lines.append(f"| {label} | {base_sect}/{n} | {lora_sect}/{n} |")
    lines.append("")
    return "\n".join(lines)


def print_results_to_stdout(
    prompt_ids: list[str],
    base_checks: list[dict],
    lora_checks: list[dict],
    n: int,
) -> None:
    base_pass = sum(1 for c in base_checks if c["all_pass"])
    lora_pass = sum(1 for c in lora_checks if c["all_pass"])

    print("\n" + "=" * 70)
    print("TEMPLATE COMPLIANCE EVALUATION")
    print("=" * 70)
    print(f"\n{'Prompt':<12} {'Base':>20} {ADAPTER_LABEL:>20}")
    print("-" * 54)
    for pid, bc, lc in zip(prompt_ids, base_checks, lora_checks):
        b_str = "PASS" if bc["all_pass"] else f"FAIL (bullets:{bc['bullet_count']})"
        l_str = "PASS" if lc["all_pass"] else f"FAIL (bullets:{lc['bullet_count']})"
        print(f"{pid:<12} {b_str:>20} {l_str:>20}")

    print("-" * 54)
    print(f"\nBase model:  {base_pass}/{n} fully compliant  ({base_pass/n*100:.0f}%)")
    print(f"{ADAPTER_LABEL} adapter: {lora_pass}/{n} fully compliant  ({lora_pass/n*100:.0f}%)")

    improvement = lora_pass - base_pass
    if improvement > 0:
        print(f"\n{ADAPTER_LABEL} improved template compliance by +{improvement} prompts.")
    elif improvement == 0:
        print(f"\nNo change in compliance between base and {ADAPTER_LABEL}.")
    else:
        print(f"\nBase outperformed {ADAPTER_LABEL} by {-improvement} prompts.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate template compliance.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate model outputs even if before_after.md already exists.",
    )
    args = parser.parse_args()

    # Resolve model ID from training metadata
    meta_path = ADAPTER_DIR / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            model_id = json.load(f)["model_id"]
    else:
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    prompts = load_eval_prompts()
    prompt_ids = [ex.get("id", f"test_{i:02d}") for i, ex in enumerate(prompts, 1)]
    n = len(prompts)

    # Obtain outputs: parse existing report or regenerate
    if not args.regenerate and BEFORE_AFTER_PATH.exists():
        print(f"Parsing existing report: {BEFORE_AFTER_PATH}")
        base_outputs, lora_outputs = parse_outputs_from_report(BEFORE_AFTER_PATH)
        if len(base_outputs) != n or len(lora_outputs) != n:
            print("Warning: report parsing yielded unexpected number of outputs; regenerating.")
            base_outputs, lora_outputs = run_inference_for_eval(model_id, prompts)
            BEFORE_AFTER_PATH.write_text(
                render_before_after_report(prompt_ids, prompts, base_outputs, lora_outputs, model_id)
            )
    else:
        print("Generating model outputs for evaluation...")
        base_outputs, lora_outputs = run_inference_for_eval(model_id, prompts)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        BEFORE_AFTER_PATH.write_text(
            render_before_after_report(prompt_ids, prompts, base_outputs, lora_outputs, model_id)
        )
        print(f"Before/after report written to: {BEFORE_AFTER_PATH}")

    # Compute compliance
    _, base_checks = compliance_score(base_outputs)
    _, lora_checks = compliance_score(lora_outputs)

    # Print to stdout
    print_results_to_stdout(prompt_ids, base_checks, lora_checks, n)

    # Write markdown report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = "\n".join([
        "# Template Compliance Evaluation",
        f"\nBase model: `{model_id}`  ",
        f"Adapter: `{ADAPTER_DIR}`\n",
        f"Prompts: `{TEST_PROMPTS_PATH}`  ",
        f"Count: {n}\n",
        render_summary(base_checks, lora_checks, n),
        "## Per-Prompt Detail",
        "",
        render_table(prompt_ids, base_checks, lora_checks),
        "",
    ])
    EVAL_RESULTS_PATH.write_text(report)
    print(f"Evaluation report written to: {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
