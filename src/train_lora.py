"""
LoRA / QLoRA fine-tuning with TRL SFTTrainer on Apple Silicon (MPS) or CUDA.

Override defaults with environment variables:
  MODEL_ID, MAX_SEQ_LEN, BATCH_SIZE, GRAD_ACC, EPOCHS, LORA_R,
  LEARNING_RATE, MAX_STEPS, OUTPUT_DIR, DATA_PATH, TRAIN_PATH, VAL_PATH, VAL_SPLIT,
  REPORT_TO, RUN_NAME, USE_QLORA (0|1), BNB_BITS (4|8)

QLoRA notes:
  - 4-bit NF4 quantization requires CUDA; falls back to 8-bit on MPS.
  - Requires: pip install bitsandbytes>=0.43.0
"""

import os
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from constants import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Hyperparameters (overridable via env vars)
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACC = int(os.getenv("GRAD_ACC", "8"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(LORA_R * 2)))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
MAX_STEPS     = int(os.getenv("MAX_STEPS", "-1"))
VAL_SPLIT     = float(os.getenv("VAL_SPLIT", "0.1"))
USE_QLORA     = os.getenv("USE_QLORA", "0") == "1"
BNB_BITS      = int(os.getenv("BNB_BITS", "4"))   # 4 or 8
REPORT_TO     = os.getenv("REPORT_TO", "none")
RUN_NAME      = os.getenv("RUN_NAME")

ROOT = Path(__file__).parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", str(ROOT / "data" / "dataset.jsonl")))
TRAIN_PATH = Path(os.getenv("TRAIN_PATH")) if os.getenv("TRAIN_PATH") else None
VAL_PATH = Path(os.getenv("VAL_PATH")) if os.getenv("VAL_PATH") else None
_default_output = ROOT / "outputs" / "lora_adapter"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(_default_output)))


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_chat(example: dict, tokenizer) -> dict:
    """Convert input/output pair to a chat-formatted string for SFT."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_datasets(tokenizer) -> tuple[Dataset, Dataset, dict]:
    """Load fixed split files when provided, otherwise split DATA_PATH locally."""
    if (TRAIN_PATH is None) ^ (VAL_PATH is None):
        raise ValueError("Set both TRAIN_PATH and VAL_PATH, or neither.")

    if TRAIN_PATH and VAL_PATH:
        train_raw = load_jsonl(TRAIN_PATH)
        val_raw = load_jsonl(VAL_PATH)
        split_meta = {
            "split_mode": "explicit",
            "data_path": None,
            "train_path": str(TRAIN_PATH),
            "val_path": str(VAL_PATH),
            "val_split": None,
        }
    else:
        raw = load_jsonl(DATA_PATH)
        formatted = [format_chat(ex, tokenizer) for ex in raw]

        random.seed(42)
        indices = list(range(len(formatted)))
        random.shuffle(indices)
        n_val = max(1, round(len(indices) * VAL_SPLIT))
        val_idx = set(indices[:n_val])
        train_data = [formatted[i] for i in range(len(formatted)) if i not in val_idx]
        val_data = [formatted[i] for i in range(len(formatted)) if i in val_idx]
        split_meta = {
            "split_mode": "random",
            "data_path": str(DATA_PATH),
            "train_path": None,
            "val_path": None,
            "val_split": VAL_SPLIT,
        }
        return Dataset.from_list(train_data), Dataset.from_list(val_data), split_meta

    train_data = [format_chat(ex, tokenizer) for ex in train_raw]
    val_data = [format_chat(ex, tokenizer) for ex in val_raw]
    return Dataset.from_list(train_data), Dataset.from_list(val_data), split_meta


def load_quantized_model(model_id: str, device: str, bits: int):
    """Load model with bitsandbytes quantization for QLoRA training."""
    try:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for QLoRA.\n"
            "Install it with: pip install bitsandbytes>=0.43.0"
        )

    if device == "mps":
        raise RuntimeError(
            "QLoRA (bitsandbytes) is not supported on Apple Silicon MPS.\n"
            "Both 4-bit and 8-bit backends fail during the backward pass on MPS.\n"
            "Use USE_QLORA=0 to run standard LoRA instead."
        )

    actual_bits = bits
    if actual_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,   # double quant saves ~0.4 bits/param
            bnb_4bit_quant_type="nf4",         # NormalFloat4 — optimal for normal dists
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading {actual_bits}-bit quantized model [{model_id}]...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # bitsandbytes manages placement
        trust_remote_code=True,
    )
    # Cast LoRA-trainable layers to float32, enable gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, actual_bits


def get_target_modules(model_id: str) -> list[str]:
    """Return LoRA target module names for known model families."""
    mid = model_id.lower()
    if "qwen" in mid:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "llama" in mid or "mistral" in mid or "gemma" in mid:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Generic fallback: only attention projections
    return ["q_proj", "v_proj"]


def main() -> None:
    device = detect_device()
    print(f"Using device: {device}")
    print(f"Model: {MODEL_ID}")
    print(f"Mode: {'QLoRA (' + str(BNB_BITS) + '-bit)' if USE_QLORA else 'LoRA'}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_dataset, val_dataset, split_meta = build_datasets(tokenizer)
    print(f"Dataset: {len(train_dataset)} train | {len(val_dataset)} val")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    actual_bits = None
    if USE_QLORA:
        model, actual_bits = load_quantized_model(MODEL_ID, device, BNB_BITS)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,  # MPS is most stable with float32
            device_map={"": device},
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # LoRA config
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=get_target_modules(MODEL_ID),
        bias="none",
    )

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # QLoRA on CUDA: use paged optimizer + bf16 for efficiency
        # LoRA / QLoRA on MPS: stay in float32
        optim="paged_adamw_32bit" if (USE_QLORA and device == "cuda") else "adamw_torch",
        fp16=False,
        bf16=USE_QLORA and device == "cuda" and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=USE_QLORA,   # saves memory during QLoRA backward pass
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        report_to=REPORT_TO,
        run_name=RUN_NAME,
        dataloader_pin_memory=False,  # Pin memory is not effective on MPS
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving adapter to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save loss log for experiment comparison
    loss_log_path = OUTPUT_DIR / "loss_log.json"
    with open(loss_log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Loss log saved to: {loss_log_path}")

    # Save training metadata for reproducibility
    meta = {
        "model_id": MODEL_ID,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "grad_acc": GRAD_ACC,
        "epochs": EPOCHS,
        "device": device,
        "target_modules": get_target_modules(MODEL_ID),
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        **split_meta,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "use_qlora": USE_QLORA,
        "bnb_bits": actual_bits if USE_QLORA else None,
        "report_to": REPORT_TO,
        "run_name": RUN_NAME,
    }
    with open(OUTPUT_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining complete.")
    print(f"Adapter weights saved to: {OUTPUT_DIR}")
    print(f"Training metadata saved to: {OUTPUT_DIR / 'training_meta.json'}")


if __name__ == "__main__":
    main()
