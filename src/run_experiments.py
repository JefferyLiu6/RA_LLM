"""
Run a configurable suite of LoRA training experiments and compare their results.

Each experiment trains with different hyperparameters, saves to its own directory
under outputs/experiments/<name>/, and is evaluated on the 10 held-out test prompts.
The base model is loaded once and all adapters are hot-swapped for evaluation,
so inference only runs the expensive model load a single time.

Usage:
  python src/run_experiments.py              # train every experiment, then eval all
  python src/run_experiments.py --eval-only  # skip training, re-evaluate existing adapters
  python src/run_experiments.py --skip-done  # skip training where adapter already exists

Adding experiments:
  Edit the EXPERIMENTS list below.  Each dict needs a unique "name" and an "env"
  dict of environment variable overrides that train_lora.py understands:
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LEARNING_RATE, EPOCHS,
    BATCH_SIZE, GRAD_ACC, MAX_SEQ_LEN, VAL_SPLIT, MODEL_ID
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from constants import SYSTEM_PROMPT  # noqa: E402
from eval_template import build_prompt, check_compliance, detect_device, generate  # noqa: E402

TEST_PROMPTS_PATH = ROOT / "data" / "test_prompts.jsonl"
EXPERIMENTS_DIR = ROOT / "outputs" / "experiments"
RESULTS_PATH = ROOT / "outputs" / "experiment_results.md"

# ---------------------------------------------------------------------------
# Experiment definitions — edit this list freely.
#
# QLoRA notes:
#   USE_QLORA=1 enables bitsandbytes quantization.
#   BNB_BITS=4  → 4-bit NF4 (CUDA only; auto-falls back to 8-bit on MPS).
#   BNB_BITS=8  → 8-bit (experimental MPS support in bitsandbytes ≥0.43).
#   Requires: pip install bitsandbytes>=0.43.0
# ---------------------------------------------------------------------------
EXPERIMENTS: list[dict] = [
    {"name": "baseline",  "env": {}},
    {"name": "rank_8",    "env": {"LORA_R": "8"}},
    {"name": "rank_32",   "env": {"LORA_R": "32"}},
    {"name": "rank_64",   "env": {"LORA_R": "64"}},
    {"name": "epochs_5",  "env": {"EPOCHS": "5"}},
    {"name": "lr_1e-4",   "env": {"LEARNING_RATE": "1e-4"}},
    {"name": "lr_5e-4",   "env": {"LEARNING_RATE": "5e-4"}},
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _is_qlora(exp: dict) -> bool:
    return exp.get("env", {}).get("USE_QLORA") == "1"


def _current_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_training(exp: dict) -> None:
    name = exp["name"]

    # QLoRA requires CUDA — bitsandbytes backward pass fails on MPS
    if _is_qlora(exp) and _current_device() == "mps":
        print(f"\n[{name}] SKIPPED — QLoRA is not supported on Apple Silicon MPS.")
        print("         bitsandbytes int8/int4 backward pass crashes on MPS.")
        print("         Run on a CUDA machine to use QLoRA experiments.")
        return

    out_dir = EXPERIMENTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "OUTPUT_DIR": str(out_dir), **exp["env"]}
    train_script = ROOT / "src" / "train_lora.py"

    print(f"\n{'='*60}")
    print(f"TRAINING  [{name}]  overrides: {exp['env'] or '(none)'}")
    print(f"Output → {out_dir}")
    print("=" * 60)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(train_script)],
        env=env,
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for '{name}' (exit code {result.returncode}). "
            "Check output above for details."
        )
    print(f"[{name}] finished in {elapsed / 60:.1f} min")


# ---------------------------------------------------------------------------
# Evaluation
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


def read_loss_summary(exp_dir: Path) -> dict:
    """Extract final train loss and val loss from the saved loss_log.json."""
    log_path = exp_dir / "loss_log.json"
    if not log_path.exists():
        return {}
    log = json.loads(log_path.read_text())
    # Walk in reverse: first entry with 'loss' (not eval) = last train step
    train_loss = next(
        (e["loss"] for e in reversed(log) if "loss" in e and "eval_loss" not in e),
        None,
    )
    val_loss = next(
        (e["eval_loss"] for e in reversed(log) if "eval_loss" in e),
        None,
    )
    return {"train_loss": train_loss, "val_loss": val_loss}


def run_all_evals(experiments: list[dict], prompts: list[dict], device: str) -> dict:
    """
    Load the base model once, hot-swap every experiment's adapter for evaluation,
    and return a results dict keyed by experiment name.
    """
    trained = [
        e for e in experiments
        if (EXPERIMENTS_DIR / e["name"] / "training_meta.json").exists()
    ]
    if not trained:
        print("No trained adapters found. Run training first.")
        return {}

    # Resolve model_id from the first available metadata file
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    for exp in trained:
        meta_path = EXPERIMENTS_DIR / exp["name"] / "training_meta.json"
        if meta_path.exists():
            model_id = json.loads(meta_path.read_text())["model_id"]
            break

    print(f"\n{'='*60}")
    print(f"EVALUATION  ({len(trained)} experiments)")
    print(f"Base model: {model_id}  |  Device: {device}")
    print("=" * 60)

    print("\nLoading base model...")
    base_model, tokenizer = load_base_model(model_id, device)

    # Load all adapters under named slots so we can hot-swap without reloading
    print("Loading adapters into named slots...")
    lora_model = None
    loaded_names: list[str] = []

    for exp in trained:
        name = exp["name"]
        adapter_dir = EXPERIMENTS_DIR / name
        try:
            if lora_model is None:
                lora_model = PeftModel.from_pretrained(
                    base_model, str(adapter_dir), adapter_name=name
                )
            else:
                lora_model.load_adapter(str(adapter_dir), adapter_name=name)
            loaded_names.append(name)
            print(f"  [{name}] loaded")
        except Exception as exc:
            print(f"  [{name}] WARNING: could not load adapter — {exc}")

    if lora_model is None:
        print("No adapters loaded successfully.")
        return {}

    # Generate base model outputs once (identical regardless of which adapter is active)
    n = len(prompts)
    print(f"\nGenerating base model outputs ({n} prompts)...")
    lora_model.disable_adapter_layers()
    base_outputs = [
        generate(lora_model, tokenizer, build_prompt(ex["input"], tokenizer), device)
        for ex in prompts
    ]
    lora_model.enable_adapter_layers()

    base_checks = [check_compliance(o) for o in base_outputs]
    base_pass = sum(1 for c in base_checks if c["all_pass"])
    print(f"  Base compliance: {base_pass}/{n} ({base_pass/n*100:.0f}%)")

    # Evaluate each adapter
    results: dict = {}
    for name in loaded_names:
        print(f"\nEvaluating [{name}]...")
        lora_model.set_adapter(name)

        lora_outputs = [
            generate(lora_model, tokenizer, build_prompt(ex["input"], tokenizer), device)
            for ex in prompts
        ]
        lora_checks = [check_compliance(o) for o in lora_outputs]
        lora_pass = sum(1 for c in lora_checks if c["all_pass"])

        exp_dir = EXPERIMENTS_DIR / name
        loss_info = read_loss_summary(exp_dir)
        meta_path = exp_dir / "training_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        results[name] = {
            "lora_compliance": lora_pass,
            "base_compliance": base_pass,
            "n": n,
            "train_loss": loss_info.get("train_loss"),
            "val_loss": loss_info.get("val_loss"),
            "lora_r": meta.get("lora_r"),
            "lora_alpha": meta.get("lora_alpha"),
            "learning_rate": meta.get("learning_rate"),
            "epochs": meta.get("epochs"),
            "train_examples": meta.get("train_examples"),
            "val_examples": meta.get("val_examples"),
        }

        tl = f"{loss_info['train_loss']:.4f}" if loss_info.get("train_loss") else "n/a"
        vl = f"{loss_info['val_loss']:.4f}"   if loss_info.get("val_loss")   else "n/a"
        print(
            f"  LoRA compliance: {lora_pass}/{n} ({lora_pass/n*100:.0f}%)  "
            f"train_loss={tl}  val_loss={vl}"
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(experiments: list[dict], results: dict) -> None:
    if not results:
        return

    n = next(iter(results.values()))["n"]
    base_pass = next(iter(results.values()))["base_compliance"]

    print(f"\n{'='*76}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*76}")
    print(
        f"{'Experiment':<14} {'R':>5} {'LR':>8} {'Ep':>3} "
        f"{'TrainL':>8} {'ValL':>8} {'LoRA%':>7} {'Base%':>7}"
    )
    print("-" * 60)

    exp_env = {e["name"]: e.get("env", {}) for e in experiments}
    for name, r in results.items():
        tl = f"{r['train_loss']:.4f}" if r["train_loss"] is not None else "   n/a"
        vl = f"{r['val_loss']:.4f}"   if r["val_loss"]   is not None else "   n/a"
        lora_pct = r["lora_compliance"] / n * 100
        base_pct = base_pass / n * 100
        lr_str = f"{r.get('learning_rate') or 2e-4:.0e}"
        print(
            f"{name:<14} {str(r.get('lora_r', '?')):>5} {lr_str:>8} "
            f"{str(r.get('epochs', '?')):>3} "
            f"{tl:>8} {vl:>8} {lora_pct:>6.0f}% {base_pct:>6.0f}%"
        )

    print(f"{'='*76}")
    best = max(results.items(), key=lambda kv: kv[1]["lora_compliance"])
    print(f"Best LoRA compliance: [{best[0]}]  {best[1]['lora_compliance']}/{n}")
    if results:
        best_val = min(
            ((k, v) for k, v in results.items() if v["val_loss"] is not None),
            key=lambda kv: kv[1]["val_loss"],
            default=None,
        )
        if best_val:
            print(f"Best val loss:        [{best_val[0]}]  {best_val[1]['val_loss']:.4f}")


def write_markdown_report(experiments: list[dict], results: dict) -> None:
    if not results:
        return

    n = next(iter(results.values()))["n"]
    base_pass = next(iter(results.values()))["base_compliance"]
    exp_env = {e["name"]: e.get("env", {}) for e in experiments}

    lines = [
        "# LoRA Experiment Results",
        "",
        "| Experiment | R | LR | Epochs | Train Loss | Val Loss | LoRA Compliance | Base Compliance |",
        "|------------|---|-----|--------|------------|----------|-----------------|-----------------|",
    ]

    for name, r in results.items():
        tl = f"{r['train_loss']:.4f}" if r["train_loss"] is not None else "n/a"
        vl = f"{r['val_loss']:.4f}"   if r["val_loss"]   is not None else "n/a"
        lora_str = f"{r['lora_compliance']}/{n} ({r['lora_compliance']/n*100:.0f}%)"
        base_str = f"{base_pass}/{n} ({base_pass/n*100:.0f}%)"
        lr_str = f"{r.get('learning_rate') or 2e-4:.0e}"
        lines.append(
            f"| {name} | {r.get('lora_r','?')} | {lr_str} | {r.get('epochs','?')} "
            f"| {tl} | {vl} | {lora_str} | {base_str} |"
        )

    lines += [
        "",
        "## Experiment Configurations",
        "",
        "| Experiment | Hyperparameter Overrides |",
        "|------------|--------------------------|",
    ]
    for name in results:
        overrides = exp_env.get(name, {})
        override_str = (
            ", ".join(f"`{k}={v}`" for k, v in overrides.items()) or "all defaults"
        )
        lines.append(f"| {name} | {override_str} |")

    lines.append("")
    RESULTS_PATH.write_text("\n".join(lines))
    print(f"\nReport written to: {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a suite of LoRA experiments."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only evaluate existing adapters.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip training for experiments whose adapter already exists.",
    )
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Training phase ----
    if not args.eval_only:
        for exp in EXPERIMENTS:
            name = exp["name"]
            already_done = (EXPERIMENTS_DIR / name / "training_meta.json").exists()
            if args.skip_done and already_done:
                print(f"[{name}] skipping (adapter exists, --skip-done set)")
                continue
            run_training(exp)
    else:
        print("--eval-only: skipping all training.")

    # ---- Evaluation phase ----
    device = detect_device()
    with open(TEST_PROMPTS_PATH) as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    results = run_all_evals(EXPERIMENTS, prompts, device)

    print_summary_table(EXPERIMENTS, results)
    write_markdown_report(EXPERIMENTS, results)


if __name__ == "__main__":
    main()
