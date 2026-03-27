"""
Evaluate and compare content quality across LoRA experiments.

eval_template.py only checks that sections exist. This script measures
the *richness* of what is inside each section:

  - Word count per section (Summary, Key Points, Limitation, Follow-up)
  - Average words per bullet point
  - Lexical diversity  (unique tokens / total tokens)
  - Follow-up question validity  (actually ends with '?')
  - Specificity signal  (technical term density vs filler words)

Outputs:
  outputs/content_metrics.md    — per-experiment averages table
  outputs/content_comparison.md — side-by-side outputs per prompt

Usage:
  python src/eval_content.py                 # generate + analyse
  python src/eval_content.py --regenerate    # force re-generation even if cache exists
  python src/eval_content.py --analyse-only  # only re-analyse cached outputs
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from constants import SYSTEM_PROMPT  # noqa: E402
from eval_template import build_prompt, detect_device, generate  # noqa: E402

TEST_PROMPTS_PATH  = ROOT / "data" / "test_prompts.jsonl"
EXPERIMENTS_DIR    = ROOT / "outputs" / "experiments"
CACHE_PATH         = ROOT / "outputs" / "content_outputs.json"
METRICS_PATH       = ROOT / "outputs" / "content_metrics.md"
COMPARISON_PATH    = ROOT / "outputs" / "content_comparison.md"

# Words that add length but not information — used to compute specificity
_FILLER = {
    "the","a","an","is","are","was","were","be","been","being",
    "it","its","this","that","these","those","and","or","but",
    "in","on","at","to","for","of","with","by","from","as",
    "can","may","also","more","such","each","their","which",
    "both","about","have","has","not","no","so","when","how",
}


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

def parse_sections(text: str) -> dict:
    """Extract text content from each template section."""
    out = {}

    m = re.search(r"Summary\s*:\s*(.*?)(?=\nKey Points\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
    out["summary"] = m.group(1).strip() if m else ""

    m = re.search(r"Key Points\s*:(.*?)(?=\nLimitation\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        out["key_points"] = re.findall(r"^\s*[-*]\s+(.+)", m.group(1), re.MULTILINE)
    else:
        out["key_points"] = []

    m = re.search(r"Limitation\s*:\s*(.*?)(?=\nFollow-up Question\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
    out["limitation"] = m.group(1).strip() if m else ""

    m = re.search(r"Follow-up Question\s*:\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
    out["followup"] = m.group(1).strip() if m else ""

    return out


# ---------------------------------------------------------------------------
# Content metrics
# ---------------------------------------------------------------------------

def _wc(s: str) -> int:
    return len(s.split()) if s.strip() else 0


def content_metrics(text: str) -> dict:
    """Return a dict of content quality signals for a single output."""
    sec = parse_sections(text)

    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    total  = len(tokens)
    unique = len(set(tokens))

    content_tokens = [t for t in tokens if t not in _FILLER]
    specificity = len(content_tokens) / total if total else 0.0

    point_lengths = [_wc(p) for p in sec["key_points"]]
    avg_point_wc  = round(sum(point_lengths) / len(point_lengths), 1) if point_lengths else 0.0

    return {
        "total_words":      total,
        "lexical_diversity": round(unique / total, 3) if total else 0.0,
        "specificity":      round(specificity, 3),
        "summary_words":    _wc(sec["summary"]),
        "n_bullets":        len(sec["key_points"]),
        "avg_point_words":  avg_point_wc,
        "limitation_words": _wc(sec["limitation"]),
        "followup_words":   _wc(sec["followup"]),
        "followup_is_q":    sec["followup"].endswith("?"),
    }


def avg_metrics(metrics_list: list[dict]) -> dict:
    """Average a list of per-prompt metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if isinstance(m[k], (int, float))]
        result[k] = round(sum(vals) / len(vals), 3) if vals else None
        # For boolean keys keep as fraction
        bool_vals = [m[k] for m in metrics_list if isinstance(m[k], bool)]
        if bool_vals:
            result[k] = round(sum(bool_vals) / len(bool_vals), 2)
    return result


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def load_base_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_all_outputs(
    experiments: list[str],
    prompts: list[dict],
    device: str,
) -> dict:
    """
    Load base model + all adapters once, generate outputs for every
    (experiment, prompt) pair.  Returns dict: {exp_name: [output, ...]}.
    """
    # Locate model_id
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    for name in experiments:
        meta_p = EXPERIMENTS_DIR / name / "training_meta.json"
        if meta_p.exists():
            model_id = json.loads(meta_p.read_text())["model_id"]
            break

    print(f"Loading base model [{model_id}] on {device}...")
    base_model, tokenizer = load_base_model(model_id, device)

    # Load all adapters
    lora_model   = None
    loaded_names: list[str] = []
    for name in experiments:
        adapter_dir = EXPERIMENTS_DIR / name
        if not adapter_dir.exists():
            print(f"  [{name}] adapter dir not found — skipping")
            continue
        try:
            if lora_model is None:
                lora_model = PeftModel.from_pretrained(
                    base_model, str(adapter_dir), adapter_name=name
                )
            else:
                lora_model.load_adapter(str(adapter_dir), adapter_name=name)
            loaded_names.append(name)
            print(f"  [{name}] adapter loaded")
        except Exception as exc:
            print(f"  [{name}] WARNING: {exc}")

    outputs: dict = {}

    # Base model (adapter disabled)
    if lora_model is not None:
        print(f"\nGenerating base outputs ({len(prompts)} prompts)...")
        lora_model.disable_adapter_layers()
        outputs["base"] = [
            generate(lora_model, tokenizer, build_prompt(ex["input"], tokenizer), device)
            for ex in prompts
        ]
        lora_model.enable_adapter_layers()

    # Each adapter
    for name in loaded_names:
        print(f"Generating [{name}] outputs ({len(prompts)} prompts)...")
        lora_model.set_adapter(name)
        outputs[name] = [
            generate(lora_model, tokenizer, build_prompt(ex["input"], tokenizer), device)
            for ex in prompts
        ]

    return outputs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(all_avg: dict[str, dict]) -> None:
    if not all_avg:
        return
    print("\n" + "=" * 90)
    print("CONTENT QUALITY METRICS  (averages across all test prompts)")
    print("=" * 90)
    hdr = (
        f"{'Model':<14} {'Words':>6} {'LexDiv':>7} {'Specif':>7} "
        f"{'SumW':>5} {'AvgPtW':>7} {'LimW':>5} {'FUW':>5} {'FU=Q?':>6}"
    )
    print(hdr)
    print("-" * 70)
    for name, m in all_avg.items():
        print(
            f"{name:<14} "
            f"{m.get('total_words', 0):>6.0f} "
            f"{m.get('lexical_diversity', 0):>7.3f} "
            f"{m.get('specificity', 0):>7.3f} "
            f"{m.get('summary_words', 0):>5.0f} "
            f"{m.get('avg_point_words', 0):>7.1f} "
            f"{m.get('limitation_words', 0):>5.0f} "
            f"{m.get('followup_words', 0):>5.0f} "
            f"{m.get('followup_is_q', 0):>6.0%}"
        )
    print("=" * 90)
    print("\nColumn guide:")
    print("  Words    total word count of the output")
    print("  LexDiv   unique tokens / total tokens  (higher = more varied vocabulary)")
    print("  Specif   content tokens / total tokens (higher = fewer filler words)")
    print("  SumW     words in the Summary sentence")
    print("  AvgPtW   average words per Key Point bullet")
    print("  LimW     words in the Limitation section")
    print("  FUW      words in the Follow-up Question")
    print("  FU=Q?    fraction of outputs where Follow-up ends with '?'")


def write_metrics_markdown(all_avg: dict[str, dict]) -> None:
    lines = [
        "# Content Quality Metrics",
        "",
        "Averages across all test prompts.",
        "",
        "| Model | Total Words | Lex Diversity | Specificity | "
        "Summary W | Avg Point W | Limitation W | Follow-up W | FU ends '?' |",
        "|-------|-------------|---------------|-------------|"
        "----------|-------------|--------------|-------------|-------------|",
    ]
    for name, m in all_avg.items():
        lines.append(
            f"| {name} "
            f"| {m.get('total_words',0):.0f} "
            f"| {m.get('lexical_diversity',0):.3f} "
            f"| {m.get('specificity',0):.3f} "
            f"| {m.get('summary_words',0):.0f} "
            f"| {m.get('avg_point_words',0):.1f} "
            f"| {m.get('limitation_words',0):.0f} "
            f"| {m.get('followup_words',0):.0f} "
            f"| {m.get('followup_is_q',0):.0%} |"
        )
    lines += [
        "",
        "**Column guide**",
        "- **Lex Diversity** — unique tokens / total tokens. Higher = more varied vocabulary.",
        "- **Specificity** — content tokens / total tokens (excl. common filler words). Higher = denser information.",
        "- **Summary W** — word count of the Summary sentence. More words often means more precise phrasing.",
        "- **Avg Point W** — average words per Key Point bullet. Higher = more detailed explanations.",
        "- **FU ends '?'** — fraction where the Follow-up Question is actually phrased as a question.",
        "",
    ]
    METRICS_PATH.write_text("\n".join(lines))
    print(f"Metrics report written to: {METRICS_PATH}")


def write_comparison_markdown(
    prompts: list[dict],
    outputs: dict[str, list[str]],
) -> None:
    """Write one section per prompt showing outputs from every model side-by-side."""
    model_names = list(outputs.keys())
    blocks = [
        "# Content Comparison: All Experiments vs Base",
        "",
        "One section per test prompt. Each model's output is shown in full.",
        "",
        "---",
        "",
    ]

    for i, ex in enumerate(prompts):
        pid = ex.get("id", f"test_{i+1:02d}")
        blocks.append(f"## {pid}")
        blocks.append("")
        blocks.append(f"**Input:** {ex['input']}")
        blocks.append("")

        for name in model_names:
            outs = outputs.get(name, [])
            text = outs[i] if i < len(outs) else "_not generated_"
            m = content_metrics(text)
            blocks.append(f"### {name}")
            blocks.append(
                f"_words: {m['total_words']} | lex_div: {m['lexical_diversity']:.3f} | "
                f"specificity: {m['specificity']:.3f}_"
            )
            blocks.append("```")
            blocks.append(text)
            blocks.append("```")
            blocks.append("")

        blocks.append("---")
        blocks.append("")

    COMPARISON_PATH.write_text("\n".join(blocks))
    print(f"Side-by-side comparison written to: {COMPARISON_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate content quality across experiments.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate model outputs even if a cache exists.",
    )
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Only re-analyse cached outputs — do not load any model.",
    )
    args = parser.parse_args()

    with open(TEST_PROMPTS_PATH) as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    # Discover trained experiments in the order they appear in experiments dir
    trained = sorted(
        [d.name for d in EXPERIMENTS_DIR.iterdir() if (d / "training_meta.json").exists()]
    ) if EXPERIMENTS_DIR.exists() else []

    if not trained:
        print("No trained experiments found in outputs/experiments/.")
        print("Run `python src/run_experiments.py` first.")
        sys.exit(1)

    # ---- Generate or load outputs ----
    if args.analyse_only:
        if not CACHE_PATH.exists():
            print(f"No cached outputs at {CACHE_PATH}. Run without --analyse-only first.")
            sys.exit(1)
        print(f"Loading cached outputs from {CACHE_PATH}")
        outputs = json.loads(CACHE_PATH.read_text())
    elif not args.regenerate and CACHE_PATH.exists():
        print(f"Using cached outputs from {CACHE_PATH}  (use --regenerate to refresh)")
        outputs = json.loads(CACHE_PATH.read_text())
        # generate for any newly trained experiments not yet in cache
        missing = [n for n in trained if n not in outputs]
        if missing:
            print(f"New experiments not in cache: {missing} — regenerating those.")
            device = detect_device()
            new_outputs = generate_all_outputs(missing, prompts, device)
            outputs.update(new_outputs)
            CACHE_PATH.write_text(json.dumps(outputs, indent=2))
    else:
        device = detect_device()
        outputs = generate_all_outputs(trained, prompts, device)
        # Cache for future --analyse-only runs
        ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(outputs, indent=2))
        print(f"Outputs cached to: {CACHE_PATH}")

    # ---- Compute metrics ----
    all_avg: dict[str, dict] = {}
    # Show base first, then experiments alphabetically
    ordered = ["base"] + sorted(k for k in outputs if k != "base")
    for name in ordered:
        if name not in outputs:
            continue
        per_prompt = [content_metrics(text) for text in outputs[name]]
        all_avg[name] = avg_metrics(per_prompt)

    print_metrics_table(all_avg)
    write_metrics_markdown(all_avg)
    write_comparison_markdown(prompts, {k: outputs[k] for k in ordered if k in outputs})


if __name__ == "__main__":
    main()
