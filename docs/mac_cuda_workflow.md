# Mac-First / CUDA-Final Workflow

This project should be developed in two passes:

1. Run everything that is reliable on Apple Silicon locally first.
2. Move only CUDA-dependent work to the RTX 4090/Linux machine.

## What Runs Where

| Work item | Mac/MPS | CUDA/RTX 4090 | Default |
|-----------|---------|---------------|---------|
| Dataset generation, validation, docs | Yes | Yes | Mac |
| Standard LoRA SFT | Yes | Yes | Mac first, CUDA final |
| Rank/LR/epoch sweeps | Yes | Yes | Mac first |
| Deterministic evals and dashboard | Yes | Yes | Mac |
| W&B experiment logging | Yes | Yes | CUDA final |
| DPO | Maybe | Yes | CUDA final |
| QLoRA with bitsandbytes | No | Yes | CUDA only |
| vLLM serving and benchmark | Not for final claim | Yes | CUDA only |
| AWQ quantization benchmark | No | Yes | CUDA only |

## Mac Pass

Use the Mac pass to prove the pipeline and iterate cheaply.

```bash
make mac-check
make mac-train
make mac-eval
```

`make mac-check` must print `mps_available=True`. If it fails, fix the local PyTorch/MPS install before running `make mac-first`; otherwise training will fall back to CPU and be too slow.

For a one-command local pass:

```bash
make mac-first
```

The Mac pass writes the standard adapter to `outputs/lora_adapter/` so the existing notebook and eval scripts continue to work.

For the experiment-level report and dashboard, run the Mac-safe sweep first:

```bash
make mac-report
```

On Mac, the QLoRA experiment is skipped automatically because bitsandbytes QLoRA is CUDA-only.

## CUDA Pass

Use the RTX 4090/Linux box for the final training and any benchmark claims.

```bash
make cuda-check
make cuda-train
make cuda-eval
```

Run QLoRA only on CUDA:

```bash
make cuda-qlora
```

Run the full sweep on CUDA when you want W&B-backed final results:

```bash
make cuda-sweep
```

The CUDA standard LoRA adapter writes to `outputs/cuda_lora_adapter/`. The QLoRA adapter writes to `outputs/cuda_qlora_adapter/`.

## Fixed Split Training

The trainer now supports explicit split files. Once the expanded dataset exists, prefer fixed split files over the current random split:

```bash
TRAIN_PATH=data/sft_train.jsonl \
VAL_PATH=data/sft_val.jsonl \
MAC_OUTPUT_DIR=outputs/lora_adapter \
make mac-train
```

On CUDA:

```bash
TRAIN_PATH=data/sft_train.jsonl \
VAL_PATH=data/sft_val.jsonl \
CUDA_OUTPUT_DIR=outputs/cuda_lora_adapter \
make cuda-train
```

If `TRAIN_PATH` and `VAL_PATH` are not set, training falls back to the existing `data/dataset.jsonl` plus `VAL_SPLIT=0.1` random split.

## Deployment Handoff

Do not use Mac numbers for the final deployment claim. Use CUDA for:

- merged model serving
- vLLM tokens/sec
- p50/p95 latency
- VRAM usage
- AWQ/INT4 comparison

After merging the final adapter into a standalone model at `outputs/merged_model/`, serve it on CUDA with:

```bash
MERGED_MODEL=outputs/merged_model make cuda-vllm-serve
```

Keep the public README claim specific: Mac proves reproducible development; CUDA proves production-style serving.
