# Dataset

This project uses supervised examples that map a technical ML paragraph to a fixed research-notes template:

```text
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

## Current State

The original 68 hand-written examples remain in `data/dataset.jsonl`. The dataset pipeline preserves them as the gold subset in `data/dataset_gold.jsonl` with metadata added.

Synthetic records can be generated locally without an API key into `data/synthetic_codex.jsonl`. They are labeled as `source=synthetic_codex` and `generation_model=codex_session` so provenance is explicit. API-generated records, if used later, go into `data/synthetic_openai.jsonl`.

## Target Dataset

Final target size: 600 SFT examples.

| Split | Target count | Purpose |
|-------|-------------:|---------|
| Train | 400 | Fine-tuning |
| Validation | 75 | Model selection |
| Test | 125 | Final held-out evaluation |

The final test set should be newly generated or manually written, not reused from prompts already used to tune README claims.

## Schema

Every row in `data/sft_all.jsonl` and split files has:

```json
{
  "id": "gold_0001",
  "input": "...",
  "output": "...",
  "source": "human_gold",
  "topic": "architectures",
  "difficulty": "intermediate",
  "split": "train",
  "input_sha256": "...",
  "generation_model": "gpt-4o-mini"
}
```

`generation_model` is only present for synthetic records.

Valid topics:

- `architectures`
- `training_optimization`
- `evaluation_metrics`
- `alignment_preference_learning`
- `systems_deployment`
- `theory_generalization`
- `data_preprocessing`
- `safety_robustness`

Valid difficulties: `intro`, `intermediate`, `advanced`.

## Pipeline Commands

Bootstrap the current 68 examples:

```bash
make dataset-bootstrap
```

Generate synthetic examples locally, no API key:

```bash
make dataset-generate
```

Generate synthetic examples with OpenAI Structured Outputs instead:

```bash
OPENAI_API_KEY=... GEN_MODEL=gpt-4o-mini make dataset-generate-openai
```

Build fixed train/val/test split files:

```bash
make dataset-build
```

Validate the full split dataset:

```bash
make dataset-validate
```

Train on the fixed split files:

```bash
TRAIN_PATH=data/sft_train.jsonl \
VAL_PATH=data/sft_val.jsonl \
make mac-train-sft
```

## Quality Gates

The validator checks:

- required metadata fields exist
- `input_sha256` matches normalized input text
- no duplicate IDs
- no duplicate inputs across splits
- topic, difficulty, and split values are valid
- output follows the exact required template
- Key Points has exactly three bullets
- input diversity is summarized with average length and repeated 4-word openings

The validator writes `outputs/dataset_report.md`.

## Manual Audit

Before using synthetic data for final training:

1. Randomly audit at least 50 synthetic examples.
2. Edit weak examples and set `source` to `manual_edit`.
3. Ensure the final test split contains no near-duplicates of train or validation inputs.
4. Keep claims in the README tied only to the final held-out test split.
