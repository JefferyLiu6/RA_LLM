"""
Run inference with base model and LoRA adapter, compare outputs.

Usage:
  # Single prompt comparison (printed to stdout)
  python src/infer.py --prompt "Describe attention mechanisms."

  # Run all 10 held-out test prompts and write outputs/before_after.md
  python src/infer.py --test-set
"""

import argparse
import json
import os
from pathlib import Path
from textwrap import indent

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
ADAPTER_DIR = ROOT / "outputs" / "lora_adapter"
TEST_PROMPTS_PATH = ROOT / "data" / "test_prompts.jsonl"
OUTPUT_DIR = ROOT / "outputs"

GEN_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))
GEN_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_id: str, device: str):
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


def load_lora_model(base_model, adapter_dir: Path):
    """Wrap a base model with the LoRA adapter."""
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model


def build_prompt(user_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt_text: str, device: str) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            do_sample=GEN_TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_test_prompts() -> list[dict]:
    with open(TEST_PROMPTS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_comparison_block(prompt_id: str, user_text: str, base_out: str, lora_out: str) -> str:
    lines = [
        f"## {prompt_id}",
        "",
        "**Input:**",
        f"> {user_text}",
        "",
        "### Base Model Output",
        "```",
        base_out,
        "```",
        "",
        "### LoRA Adapter Output",
        "```",
        lora_out,
        "```",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def print_comparison(prompt_id: str, user_text: str, base_out: str, lora_out: str) -> None:
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"PROMPT [{prompt_id}]: {user_text[:120]}{'...' if len(user_text) > 120 else ''}")
    print(f"{separator}")
    print("\n--- BASE MODEL ---")
    print(base_out)
    print("\n--- LoRA ADAPTER ---")
    print(lora_out)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs LoRA adapter outputs.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt text.")
    parser.add_argument(
        "--test-set",
        action="store_true",
        help="Run all 10 held-out test prompts and write outputs/before_after.md.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override base model ID (default: read from adapter training_meta.json).",
    )
    args = parser.parse_args()

    if not args.prompt and not args.test_set:
        parser.error("Provide either --prompt TEXT or --test-set.")

    # Resolve model ID
    meta_path = ADAPTER_DIR / "training_meta.json"
    if args.model_id:
        model_id = args.model_id
    elif meta_path.exists():
        with open(meta_path) as f:
            model_id = json.load(f)["model_id"]
        print(f"Using model from training metadata: {model_id}")
    else:
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"No training metadata found; defaulting to {model_id}")

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}\n"
            "Run `python src/train_lora.py` first."
        )

    device = detect_device()
    print(f"Device: {device}")

    # Load base model once; wrap it for lora
    print(f"Loading base model {model_id}...")
    base_model, tokenizer = load_model_and_tokenizer(model_id, device)

    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    lora_model = load_lora_model(base_model, ADAPTER_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test_set:
        prompts = load_test_prompts()
        print(f"\nRunning {len(prompts)} held-out test prompts...")
        report_blocks = [
            "# Before vs After: Base Model vs LoRA Adapter\n",
            f"Base model: `{model_id}`  \n",
            f"Adapter: `{ADAPTER_DIR}`\n\n---\n",
        ]

        for ex in prompts:
            prompt_id = ex.get("id", "unknown")
            user_text = ex["input"]
            prompt_str = build_prompt(user_text, tokenizer)

            # Base model: temporarily disable adapter
            lora_model.disable_adapter_layers()
            base_out = generate(lora_model, tokenizer, prompt_str, device)
            lora_model.enable_adapter_layers()

            # LoRA model
            lora_out = generate(lora_model, tokenizer, prompt_str, device)

            print_comparison(prompt_id, user_text, base_out, lora_out)
            report_blocks.append(format_comparison_block(prompt_id, user_text, base_out, lora_out))

        report_path = OUTPUT_DIR / "before_after.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_blocks))
        print(f"\nReport written to: {report_path}")

    else:
        user_text = args.prompt
        prompt_str = build_prompt(user_text, tokenizer)

        # Base model
        lora_model.disable_adapter_layers()
        base_out = generate(lora_model, tokenizer, prompt_str, device)
        lora_model.enable_adapter_layers()

        # LoRA model
        lora_out = generate(lora_model, tokenizer, prompt_str, device)

        print_comparison("custom", user_text, base_out, lora_out)


if __name__ == "__main__":
    main()
