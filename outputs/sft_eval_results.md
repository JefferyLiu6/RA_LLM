# Template Compliance Evaluation

Base model: `Qwen/Qwen2.5-1.5B-Instruct`  
Adapter: `outputs/lora_adapter`

Prompts: `data/sft_test.jsonl`  
Count: 125

## Summary

| Model | Fully Compliant | Score |
|-------|----------------|-------|
| Base  | 48/125            | 38%   |
| LoRA  | 124/125            | 99%   |

## Per-Section Compliance

| Section | Base | LoRA |
|---------|------|------|
| Summary | 64/125 | 125/125 |
| Key Points | 64/125 | 125/125 |
| 3 Bullets | 84/125 | 125/125 |
| Limitation | 64/125 | 125/125 |
| Follow-up Q | 64/125 | 124/125 |

## Per-Prompt Detail

| Prompt | Base Summary | Base Key Points | Base 3 Bullets | Base Limitation | Base Follow-up Q | Base PASS | LoRA Summary | LoRA Key Points | LoRA 3 Bullets | LoRA Limitation | LoRA Follow-up Q | LoRA PASS |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| codex_0469 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0236 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0247 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0198 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0168 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0372 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0528 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0414 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0368 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0356 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0065 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0230 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0496 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0485 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0140 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0489 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0029 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0353 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0412 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0393 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0131 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0468 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0342 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0493 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0273 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0384 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0073 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0195 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0405 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0379 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0219 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0126 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0038 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0122 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0199 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0117 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0421 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0371 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0043 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0082 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0242 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0513 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0364 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0333 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0058 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0436 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0095 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0020 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0116 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0463 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0123 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0089 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0113 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0308 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0005 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0194 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0441 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0374 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | N | N |
| codex_0458 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0086 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0500 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0066 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0094 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0470 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0401 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0429 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0190 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0182 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0375 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0292 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0471 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0531 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0271 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0159 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0305 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0283 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0031 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0267 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0143 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0334 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0141 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0057 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0225 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0042 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0275 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0330 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0310 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0408 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0461 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0298 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0286 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0367 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0041 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0417 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0243 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0017 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0176 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0270 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0015 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0383 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0244 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0260 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0363 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0085 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0107 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0302 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0416 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0505 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0185 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0193 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0137 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0220 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0059 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0338 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0157 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0209 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0529 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0229 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0322 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0217 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0158 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0507 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0450 | N | N | Y | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0022 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
| codex_0491 | N | N | N | N | N | N | Y | Y | Y | Y | Y | Y |
