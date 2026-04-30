"""
DPO preference tuning on top of the SFT LoRA adapter.

The policy and reference adapters both start from SFT_ADAPTER_DIR. During DPO,
the policy adapter is trainable and the reference adapter is frozen, so the DPO
objective compares against the SFT model rather than the raw base model.

Override defaults with environment variables:
  MODEL_ID, SFT_ADAPTER_DIR, DPO_DATA_PATH, OUTPUT_DIR,
  MAX_SEQ_LEN, MAX_PROMPT_LEN, MAX_TARGET_LEN,
  BATCH_SIZE, GRAD_ACC, EPOCHS, MAX_STEPS, LEARNING_RATE,
  DPO_BETA, REPORT_TO, RUN_NAME, EVAL_STRATEGY, SAVE_STRATEGY, DPO_LIMIT,
  REQUIRE_DEVICE, DRY_RUN
"""

import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from constants import SYSTEM_PROMPT


ROOT = Path(__file__).parent.parent

MODEL_ID = os.getenv("MODEL_ID")
SFT_ADAPTER_DIR = Path(os.getenv("SFT_ADAPTER_DIR", str(ROOT / "outputs" / "lora_adapter")))
DPO_DATA_PATH = Path(os.getenv("DPO_DATA_PATH", str(ROOT / "data" / "dpo_pairs.jsonl")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT / "outputs" / "dpo_adapter")))

MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
MAX_PROMPT_LEN = int(os.getenv("MAX_PROMPT_LEN", "256"))
MAX_TARGET_LEN = int(os.getenv("MAX_TARGET_LEN", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACC = int(os.getenv("GRAD_ACC", "4"))
EPOCHS = float(os.getenv("EPOCHS", "1"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "-1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-6"))
DPO_BETA = float(os.getenv("DPO_BETA", "0.1"))
REPORT_TO = os.getenv("REPORT_TO", "none")
RUN_NAME = os.getenv("RUN_NAME", "dpo")
EVAL_STRATEGY = os.getenv("EVAL_STRATEGY", "epoch")
SAVE_STRATEGY = os.getenv("SAVE_STRATEGY", "epoch")
DPO_LIMIT = int(os.getenv("DPO_LIMIT", "0"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
REQUIRE_DEVICE = os.getenv("REQUIRE_DEVICE")

POLICY_ADAPTER = "default"
REFERENCE_ADAPTER = "reference"


class CompatDPOTrainer(DPOTrainer):
    """Compatibility shim for TRL 0.10.x with newer Transformers Trainer APIs."""

    def get_batch_samples(self, epoch_iterator, num_batches: int, device=None):
        batch_samples = []
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        num_items_in_batch = None
        if device is not None:
            num_items_in_batch = self._get_num_items_in_batch(batch_samples, device)
        return batch_samples, num_items_in_batch

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def log(self, logs: dict[str, float], start_time=None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        stored_metrics = self._stored_metrics.get(train_eval, {})
        for key, metrics in stored_metrics.items():
            logs[key] = torch.tensor(metrics).mean().item()
        if train_eval in self._stored_metrics:
            del self._stored_metrics[train_eval]
        return super(DPOTrainer, self).log(logs, start_time)


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_model_id(adapter_dir: Path) -> str:
    if MODEL_ID:
        return MODEL_ID
    meta_path = adapter_dir / "training_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())["model_id"]
    return "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


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


def build_datasets(tokenizer) -> tuple[Dataset, Dataset, dict]:
    records = load_jsonl(DPO_DATA_PATH)
    if DPO_LIMIT > 0:
        records = records[:DPO_LIMIT]

    test_records = [record["id"] for record in records if record.get("source_split") == "test"]
    if test_records:
        raise ValueError(f"DPO data must not include held-out test records. Example ids: {test_records[:5]}")

    formatted = []
    for record in records:
        formatted.append({
            "prompt": build_prompt(record["prompt"], tokenizer),
            "chosen": record["chosen"].strip(),
            "rejected": record["rejected"].strip(),
            "id": record["id"],
            "source_id": record.get("source_id", ""),
            "source_split": record.get("source_split", ""),
            "topic": record.get("topic", ""),
            "difficulty": record.get("difficulty", ""),
        })

    train_data = [record for record in formatted if record["source_split"] == "train"]
    eval_data = [record for record in formatted if record["source_split"] == "val"]

    if not train_data:
        raise ValueError("No train records found in DPO data.")
    if not eval_data:
        eval_data = train_data[: max(1, round(len(train_data) * 0.1))]

    split_meta = {
        "dpo_data_path": str(DPO_DATA_PATH),
        "dpo_limit": DPO_LIMIT if DPO_LIMIT > 0 else None,
        "train_examples": len(train_data),
        "eval_examples": len(eval_data),
    }
    return Dataset.from_list(train_data), Dataset.from_list(eval_data), split_meta


def load_policy_with_reference(model_id: str, device: str):
    if not SFT_ADAPTER_DIR.exists():
        raise FileNotFoundError(f"SFT adapter not found: {SFT_ADAPTER_DIR}")

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"Loading base model [{model_id}] on {device} ({dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )

    print(f"Loading trainable policy adapter from {SFT_ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(SFT_ADAPTER_DIR),
        adapter_name=POLICY_ADAPTER,
        is_trainable=True,
    )

    print("Loading frozen reference adapter from the same SFT checkpoint...")
    model.load_adapter(
        str(SFT_ADAPTER_DIR),
        adapter_name=REFERENCE_ADAPTER,
        is_trainable=False,
    )
    model.set_adapter(POLICY_ADAPTER)
    model.print_trainable_parameters()
    return model, dtype


def load_tokenizer(model_id: str):
    tokenizer_source: str | Path = model_id
    if (SFT_ADAPTER_DIR / "tokenizer_config.json").exists():
        tokenizer_source = SFT_ADAPTER_DIR

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main() -> None:
    device = detect_device()
    if REQUIRE_DEVICE and device != REQUIRE_DEVICE:
        raise RuntimeError(f"Required device '{REQUIRE_DEVICE}' is not available; detected '{device}'.")

    model_id = resolve_model_id(SFT_ADAPTER_DIR)

    print(f"Using device: {device}")
    print(f"Model: {model_id}")
    print(f"SFT adapter: {SFT_ADAPTER_DIR}")
    print(f"DPO data: {DPO_DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_id)

    train_dataset, eval_dataset, split_meta = build_datasets(tokenizer)
    print(f"Dataset: {len(train_dataset)} train | {len(eval_dataset)} eval")

    if DRY_RUN:
        print("DRY_RUN=1 set; dataset/tokenizer checks passed, skipping model load and training.")
        return

    model, dtype = load_policy_with_reference(model_id, device)

    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1 if MAX_STEPS > 0 else 5,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=2,
        optim="adamw_torch",
        fp16=False,
        bf16=device == "cuda" and dtype == torch.bfloat16,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        max_target_length=MAX_TARGET_LEN,
        beta=DPO_BETA,
        model_adapter_name=POLICY_ADAPTER,
        ref_adapter_name=REFERENCE_ADAPTER,
        remove_unused_columns=False,
        report_to=REPORT_TO,
        run_name=RUN_NAME,
        dataloader_pin_memory=False,
    )

    trainer = CompatDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if EVAL_STRATEGY != "no" else None,
        tokenizer=tokenizer,
    )

    print("\nStarting DPO training...")
    trainer.train()

    print(f"\nSaving DPO policy adapter to {OUTPUT_DIR}")
    trainer.model.set_adapter(POLICY_ADAPTER)
    trainer.model.save_pretrained(str(OUTPUT_DIR), selected_adapters=[POLICY_ADAPTER])
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    loss_log_path = OUTPUT_DIR / "dpo_loss_log.json"
    with loss_log_path.open("w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"DPO loss log saved to: {loss_log_path}")

    meta = {
        "model_id": model_id,
        "sft_adapter_dir": str(SFT_ADAPTER_DIR),
        "output_dir": str(OUTPUT_DIR),
        "device": device,
        "max_seq_len": MAX_SEQ_LEN,
        "max_prompt_len": MAX_PROMPT_LEN,
        "max_target_len": MAX_TARGET_LEN,
        "batch_size": BATCH_SIZE,
        "grad_acc": GRAD_ACC,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "dpo_beta": DPO_BETA,
        "policy_adapter": POLICY_ADAPTER,
        "reference_adapter": REFERENCE_ADAPTER,
        "report_to": REPORT_TO,
        "run_name": RUN_NAME,
        **split_meta,
    }
    with (OUTPUT_DIR / "dpo_training_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("\nDPO training complete.")
    print(f"Adapter weights saved to: {OUTPUT_DIR}")
    print(f"DPO metadata saved to: {OUTPUT_DIR / 'dpo_training_meta.json'}")


if __name__ == "__main__":
    main()
