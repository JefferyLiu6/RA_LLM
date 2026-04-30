# Template Compliance Evaluation

Base model: `Qwen/Qwen2.5-1.5B-Instruct`  
Adapter: `outputs/lora_adapter`

Prompts: `data/sft_test.jsonl`  
Count: 10

## Summary

| Model | Fully Compliant | Score |
|-------|----------------|-------|
| Base  | 4/10            | 40%   |
| LoRA  | 10/10            | 100%   |

## Per-Section Compliance

| Section | Base | LoRA |
|---------|------|------|
| Summary | 5/10 | 10/10 |
| Key Points | 5/10 | 10/10 |
| 3 Bullets | 5/10 | 10/10 |
| Limitation | 5/10 | 10/10 |
| Follow-up Q | 5/10 | 10/10 |

## Per-Prompt Detail

| Prompt | Base Summary | Base Key Points | Base 3 Bullets | Base Limitation | Base Follow-up Q | Base PASS | LoRA Summary | LoRA Key Points | LoRA 3 Bullets | LoRA Limitation | LoRA Follow-up Q | LoRA PASS |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| codex_0469 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0236 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0247 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0198 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0168 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0372 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0528 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0414 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0368 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0356 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
