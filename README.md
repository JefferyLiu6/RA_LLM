# LoRA Research Notes Adapter

A weekend LoRA project that teaches a small instruct model to convert ML technical text into structured research notes.

## How LoRA Works

Standard fine-tuning updates every weight in the model. LoRA instead **freezes** all pretrained weights and injects a pair of small trainable matrices (A and B) into each transformer layer. The weight update is expressed as a low-rank product `ΔW = B × A`, where the rank `r` is much smaller than the original weight dimensions.

```mermaid
flowchart LR
    subgraph loraPath ["Low-rank path  (trained)"]
        A["Matrix A\n(r × d_in)"]
        B["Matrix B\n(d_out × r)"]
        A -->|"rank r << d"| B
    end

    subgraph frozenPath ["Original weight  (frozen)"]
        W["Pretrained W\n(d_out × d_in)"]
    end

    InputX["Input  x"] --> W
    InputX --> A
    W --> AddNode["⊕  add"]
    B --> AddNode
    AddNode --> OutputH["Output  h = Wx + BAx"]
```



### What `r` (rank) means

`r` is the single number that controls how much the adapter can learn. It is the width of the bottleneck between matrices A and B.

```mermaid
flowchart LR
    X2["d_in = 1536\n(model hidden size)"]
    A2["Matrix A\n1536 × r"]
    B2["Matrix B\nr × 1536"]
    Y2["d_out = 1536"]
    X2 -->|"compress"| A2 -->|"r dimensions"| B2 -->|"expand"| Y2
```



Concretely for `Qwen2.5-1.5B` with `d = 1536`:


| r      | Params per weight pair   | Total adapter params | What it captures                    |
| ------ | ------------------------ | -------------------- | ----------------------------------- |
| 4      | 2 × 1536 × 4 = 12 K      | ~2 M                 | Very simple shifts — often too few  |
| 8      | 2 × 1536 × 8 = 25 K      | ~4 M                 | Basic format learning               |
| **16** | **2 × 1536 × 16 = 49 K** | **~10 M**            | **Good default — used in baseline** |
| 32     | 2 × 1536 × 32 = 98 K     | ~19 M                | Richer style changes                |
| 64     | 2 × 1536 × 64 = 197 K    | ~38 M                | Best compliance in sweep (100%)     |


Think of `r` as the number of "directions" the adapter is allowed to steer the model's representations. A low `r` forces the adapter to learn only the most essential transformations. A high `r` gives more flexibility but risks overfitting on small datasets and uses more memory.

The rank doesn't need to be large because output-format learning is a low-complexity task — the model already knows how to write; it just needs a small nudge to follow a specific template consistently.

**Why this works well:**

- Only `r × (d_in + d_out)` parameters are trained per layer instead of `d_in × d_out`
- At inference the adapter can be **merged** into W with zero added latency
- A rank of 16–64 is typically enough to teach a new output format

---

## LoRA vs Full Fine-Tuning

```mermaid
flowchart TB
    subgraph fullFT ["Full Fine-Tuning  —  ~1.5 B trainable params  (100%)"]
        direction TB
        FE["Embedding\nupdated"]
        FA["x28  Attention blocks\nWq  Wk  Wv  Wo\nupdated"]
        FF["x28  FFN blocks\nWgate  Wup  Wdown\nupdated"]
        FH["LM Head\nupdated"]
        FE --> FA --> FF --> FH
    end

    subgraph loraFT ["LoRA  r=16  —  ~10 M trainable params  (0.7%)"]
        direction TB
        LE["Embedding\nfrozen"]
        LA["x28  Attention blocks\nWq  Wk  Wv  Wo  frozen\nAq Bq  Av Bv  trained"]
        LF["x28  FFN blocks\nWgate  Wup  Wdown  frozen\nAg Bg  Au Bu  Ad Bd  trained"]
        LH["LM Head\nfrozen"]
        LE --> LA --> LF --> LH
    end
```




| Property                | Full Fine-Tuning            | LoRA (r=16)                        |
| ----------------------- | --------------------------- | ---------------------------------- |
| Trainable params        | ~1,500 M (100%)             | ~10 M (0.7%)                       |
| Optimizer memory        | Full copy of all weights    | Only A + B matrices                |
| VRAM needed             | Very high (often multi-GPU) | Low (fits on Apple Silicon)        |
| Training time           | Hours to days               | Minutes to ~30 min                 |
| Catastrophic forgetting | High risk                   | Low — backbone is frozen           |
| Inference cost          | Same as base                | Zero overhead if adapter is merged |
| Portability             | Entire new model checkpoint | Tiny adapter file (~40 MB at r=16) |


LoRA's frozen backbone is the key reason this project runs on a MacBook: the optimizer only needs to hold gradient states for the ~10 M adapter parameters rather than for all 1.5 B weights.

---

## Project Workflow

```mermaid
flowchart TD
    subgraph dataPrep ["1 · Data Preparation"]
        DS["dataset.jsonl\n68 training examples"]
        TP["test_prompts.jsonl\n10 held-out prompts"]
        FMT["format_chat\nWrap in chat template\nsystem + user + assistant"]
        DS --> FMT
    end

    subgraph training ["2 · Training"]
        BASE["Qwen2.5-1.5B-Instruct\nfrozen backbone"]
        LORA["LoraConfig\nr · alpha · target_modules"]
        SFT["SFTTrainer  TRL\nCausalLM · MPS · fp32"]
        FMT --> SFT
        BASE --> SFT
        LORA --> SFT
        SFT --> ADAPTER["outputs/lora_adapter/\nadapter_model.safetensors"]
    end

    subgraph inference ["3 · Inference"]
        INFER["infer.py\nload base  →  disable adapter  →  generate\nload base  →  enable adapter   →  generate"]
        ADAPTER --> INFER
        BASE --> INFER
        TP --> INFER
        INFER --> REPORT["outputs/before_after.md"]
    end

    subgraph evaluation ["4 · Evaluation"]
        EVAL["eval_template.py\nCheck each output for:\n· Summary present\n· Key Points present\n· Exactly 3 bullets\n· Limitation present\n· Follow-up Question present"]
        REPORT --> EVAL
        EVAL --> SCORES["outputs/eval_results.md\nBase: 10%  →  LoRA: 70–100%"]
    end
```



---

## Output Template

Given any ML concept, paragraph, or abstract, the tuned model outputs:

```
Summary:
<one-sentence summary>

Key Points:
- <point 1>
- <point 2>
- <point 3>

Limitation:
<one key limitation>

Follow-up Question:
<one question worth exploring>
```

---

## Stack


| Component   | Choice                       |
| ----------- | ---------------------------- |
| Base model  | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fine-tuning | LoRA via PEFT                |
| Trainer     | SFTTrainer (TRL)             |
| Device      | Apple Silicon MPS            |


---

## Project Layout

```
research_ass/
├── data/
│   ├── dataset.jsonl        # 68 training examples
│   └── test_prompts.jsonl   # 10 held-out prompts
├── src/
│   ├── train_lora.py        # SFTTrainer entrypoint
│   ├── infer.py             # Base vs LoRA inference comparison
│   └── eval_template.py     # Format-adherence evaluator
├── outputs/
│   ├── lora_adapter/        # Saved adapter weights
│   ├── before_after.md      # Per-prompt base vs LoRA comparison
│   ├── eval_results.md      # Format compliance scores
│   ├── experiment_results.md
│   ├── content_comparison.md
│   └── content_metrics.md
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Python 3.10+ required
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> First run downloads the base model (~3 GB). Ensure you are on WiFi.

---

## Train

```bash
python src/train_lora.py
```

Adapter is saved to `outputs/lora_adapter/`. Training takes **10–30 minutes** on Apple Silicon M-series depending on chip generation.

Key flags overridable via environment variables:


| Variable      | Default                      | Effect                           |
| ------------- | ---------------------------- | -------------------------------- |
| `MODEL_ID`    | `Qwen/Qwen2.5-1.5B-Instruct` | Base model HF ID                 |
| `MAX_SEQ_LEN` | `512`                        | Token sequence length            |
| `BATCH_SIZE`  | `2`                          | Per-device batch size            |
| `GRAD_ACC`    | `8`                          | Gradient accumulation steps      |
| `EPOCHS`      | `3`                          | Training epochs                  |
| `LORA_R`      | `16`                         | LoRA rank                        |
| `VAL_SPLIT`   | `0.1`                        | Fraction held out for validation |


Memory-constrained example:

```bash
MAX_SEQ_LEN=256 BATCH_SIZE=1 GRAD_ACC=16 python src/train_lora.py
```

---

## Inference: Before vs After

```bash
# Single prompt
python src/infer.py --prompt "Attention is a mechanism in neural networks that assigns weights to input tokens."

# All 10 held-out prompts → outputs/before_after.md
python src/infer.py --test-set
```

---

## Evaluate

```bash
python src/eval_template.py
```

Checks each output for all required sections and exactly 3 key-point bullets. Writes `outputs/eval_results.md`.

---

## Experiment Results

The screenshot below shows the same prompt (`test_01 — softmax`) run through three configurations side by side: the untuned base model, a low-rank adapter (r=8), and a high-rank adapter (r=64).

![Rank comparison: No Adaptation vs LoRA Low Rank vs LoRA High Rank](assets/screenshots/rank_comparison.png)

- **No Adaptation (0% compliant)** — uses `**bold**` markdown headers instead of plain `Header:`, and produces 4 bullets instead of 3
- **LoRA Low Rank r=8 (55% compliant)** — correct plain headers, correct section order, but only 2 bullets in Key Points
- **LoRA High Rank r=64 (91% compliant)** — correct plain headers, exactly 3 bullets, proper Limitation and Follow-up Question

Results from the hyperparameter sweep conducted after training:


| Experiment | R   | LR   | Epochs | Train Loss | Val Loss | LoRA Compliance  | Base Compliance |
| ---------- | --- | ---- | ------ | ---------- | -------- | ---------------- | --------------- |
| baseline   | 16  | 2e-4 | 3      | 1.747      | 1.589    | 9/10 (90%)       | 1/10 (10%)      |
| rank_8     | 8   | 2e-4 | 3      | 2.014      | 1.874    | 7/10 (70%)       | 1/10 (10%)      |
| rank_32    | 32  | 2e-4 | 3      | 1.495      | 1.451    | 9/10 (90%)       | 1/10 (10%)      |
| rank_64    | 64  | 2e-4 | 3      | 1.331      | 1.422    | **10/10 (100%)** | 1/10 (10%)      |
| epochs_5   | 16  | 2e-4 | 5      | 1.359      | 1.435    | **10/10 (100%)** | 1/10 (10%)      |
| lr_1e-4    | 16  | 1e-4 | 3      | 2.057      | 1.939    | 7/10 (70%)       | 1/10 (10%)      |
| lr_5e-4    | 16  | 5e-4 | 3      | 1.413      | 1.437    | 9/10 (90%)       | 1/10 (10%)      |


**Key takeaways:**

- `rank_64` and `epochs_5` both achieve 100% compliance with no overfitting (val loss tracks train loss)
- `lr=1e-4` is too conservative — higher loss, lower compliance
- `rank_8` is the practical floor; acceptable but noticeably weaker

---

## Troubleshooting (MPS)

**First run is slow** — MPS compiles Metal shaders on first use; subsequent runs are faster.

**OOM / killed process** — Lower `MAX_SEQ_LEN` (try 256) and increase `GRAD_ACC`. Set `BATCH_SIZE=1`.

`**NotImplementedError: mps`** — Add the fallback flag:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_lora.py
```

**Adapter not loading** — Use the same `MODEL_ID` for training and inference. The correct ID is stored in `outputs/lora_adapter/training_meta.json`.

---

## What Success Looks Like


| Model                        | Format Compliance |
| ---------------------------- | ----------------- |
| Base `Qwen2.5-1.5B-Instruct` | 10% (1/10)        |
| LoRA r=8                     | 70%               |
| LoRA r=16 baseline           | 90%               |
| LoRA r=64 or 5 epochs        | **100%**          |


The base model's failure modes are: wrong bullet count (4–5 instead of 3), `###` markdown headers instead of plain `Header:`, and `**bold`** formatting. The LoRA adapter reliably fixes all three on unseen prompts.

---

## Before / After: Why LoRA Produces Better Output

![Before / After — No Adaptation vs LoRA Low Rank vs LoRA High Rank](assets/screenshots/before_after_chat.png)

The same prompt sent to three versions of the model reveals exactly what fine-tuning teaches:

**No Adaptation (Base) — 0% compliant**
The base model has never seen the required output format. It defaults to its general instruction-following behaviour: wrapping section names in `**bold markdown`**, producing a variable number of bullets (3–5), and writing a vague limitation like `- Sensitive to the scale of input data`. None of this matches the target template, so it fails every automated check.

**LoRA — Low Rank (r=8) — ~55–70% compliant**
With only a small rank-8 adapter (~1 M trainable parameters), the model has partially learned the format. Section headers are now plain (`Summary:`, `Key Points:`) with no markdown formatting. However, r=8 limits the adapter's expressiveness: it sometimes produces only 2 bullets instead of 3, or drops a section entirely. The format signal is there, but capacity is not sufficient to apply it consistently across all prompt styles.

**LoRA — High Rank (r=64) — ~91–100% compliant**
At r=64 the adapter has enough capacity to memorise the exact template and generalise it to unseen prompts. Every output has exactly 3 bullets, plain `Section:` headers, a substantive one-sentence limitation, and a genuine follow-up question. The improvement is entirely structural — the base weights are frozen, so all changes come from the ~8 M parameters in the injected A and B matrices.

The core lesson: LoRA does not change what the model *knows* — it changes how the model *formats* what it knows. A well-configured adapter (rank ≥ 32, ≥ 3 epochs) is sufficient to teach a reliable output schema with under 1% of the model's total parameters.