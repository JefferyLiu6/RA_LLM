# Template Compliance Evaluation

Base model: `Qwen/Qwen2.5-1.5B-Instruct`  
Adapter: `outputs/dpo_adapter`

Prompts: `data/sft_test.jsonl`  
Count: 125

## Summary

| Model | Fully Compliant | Score |
|-------|----------------|-------|
| Base  | 105/125            | 84%   |
| DPO  | 125/125            | 100%   |

## Per-Section Compliance

| Section | Base | DPO |
|---------|------|------|
| Summary | 125/125 | 125/125 |
| Key Points | 125/125 | 125/125 |
| 3 Bullets | 105/125 | 125/125 |
| Limitation | 125/125 | 125/125 |
| Follow-up Q | 125/125 | 125/125 |

## Per-Prompt Detail

| Prompt | Base Summary | Base Key Points | Base 3 Bullets | Base Limitation | Base Follow-up Q | Base PASS | DPO Summary | DPO Key Points | DPO 3 Bullets | DPO Limitation | DPO Follow-up Q | DPO PASS |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| codex_0469 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0236 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0247 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0198 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0168 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0372 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0528 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0414 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0368 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0356 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0065 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0230 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0496 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0485 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0140 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0489 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0029 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0353 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0412 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0393 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0131 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0468 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0342 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0493 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0273 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0384 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0073 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0195 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0405 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0379 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0219 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0126 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0038 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0122 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0199 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0117 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0421 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0371 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0043 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0082 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0242 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0513 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0364 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0333 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0058 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0436 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0095 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0020 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0116 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0463 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0123 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0089 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0113 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0308 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0005 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0194 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0441 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0374 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0458 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0086 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0500 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0066 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0094 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0470 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0401 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0429 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0190 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0182 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0375 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0292 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0471 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0531 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0271 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0159 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0305 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0283 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0031 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0267 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0143 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0334 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0141 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0057 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0225 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0042 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0275 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0330 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0310 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0408 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0461 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0298 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0286 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0367 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0041 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0417 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0243 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0017 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0176 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0270 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0015 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0383 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0244 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0260 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0363 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0085 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0107 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0302 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0416 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0505 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0185 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0193 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0137 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0220 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0059 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0338 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0157 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0209 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0529 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0229 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0322 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0217 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0158 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0507 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0450 | Y | Y | N | Y | Y | N | Y | Y | Y | Y | Y | Y |
| codex_0022 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| codex_0491 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
