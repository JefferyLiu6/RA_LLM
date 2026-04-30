# Template Compliance Evaluation

Base model: `Qwen/Qwen2.5-1.5B-Instruct`  
Adapter: `outputs/dpo_adapter`

Prompts: `data/sft_test.jsonl`  
Count: 2

## Summary

| Model | Fully Compliant | Score |
|-------|----------------|-------|
| Base  | 2/2            | 100%   |
| LoRA  | 2/2            | 100%   |

## Per-Section Compliance

| Section | Base | LoRA |
|---------|------|------|
| Summary | 2/2 | 2/2 |
| Key Points | 2/2 | 2/2 |
| 3 Bullets | 2/2 | 2/2 |
| Limitation | 2/2 | 2/2 |
| Follow-up Q | 2/2 | 2/2 |

## Per-Prompt Detail

| Prompt | Base Summary | Base Key Points | Base 3 Bullets | Base Limitation | Base Follow-up Q | Base PASS | LoRA Summary | LoRA Key Points | LoRA 3 Bullets | LoRA Limitation | LoRA Follow-up Q | LoRA PASS |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| codex_0374 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0375 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
