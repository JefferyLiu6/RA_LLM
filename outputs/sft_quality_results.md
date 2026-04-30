# SFT Quality Evaluation

References: `data/sft_test.jsonl`  
Cached generations: `outputs/sft_before_after.md`  
Examples: 125

## Summary

The content metrics compare generated outputs against the held-out reference answer for the same prompt. `Content Alignment` is a heuristic weighted score: 60% reference-token F1, 25% key-term recall, and 15% supported-token rate.

| Model | Format Compliance | Content Alignment | Ref Token F1 | Key-Term Recall | Unsupported Terms | Follow-up Valid | Formulaic Rate | Grammar Flags | Avg Words |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| Base | 38% | 29% | 27% | 26% | 60% | 51% | 0% | 0% | 101.9 |
| LoRA | 99% | 86% | 86% | 84% | 11% | 99% | 99% | 2% | 123.8 |

## Interpretation

- `Format Compliance` is the strict template check: exact section headers, exactly three bullets, and a follow-up that ends with `?`.
- `Content Alignment` is not human judgment; it checks overlap with the held-out reference answer and whether generated terms are supported by the prompt/reference.
- `Formulaic Rate` is a warning signal for repeated learned phrasing. High format compliance with high formulaic rate usually means the next improvement should be DPO or style-diverse SFT data.

## LoRA Failure / Risk Examples

| Prompt | Issue | Follow-up / Opening |
|--------|-------|---------------------|
| codex_0236 | plural_subject_uses, how_should_ranks | How should sparse MoE ranks tokens before sending them to experts? |
| codex_0374 | format, extra_sentence_after_question | How much self-critique is too much? Teams often set a limit based on qualitative feedback rather than a hard number. |

## Lowest LoRA Content-Alignment Examples

| Prompt | Topic | Difficulty | Content Alignment | Ref Token F1 | Key-Term Recall |
|--------|-------|------------|:---:|:---:|:---:|
| codex_0217 | systems_deployment | intermediate | 49% | 50% | 42% |
| codex_0198 | safety_robustness | intermediate | 50% | 49% | 45% |
| codex_0371 | safety_robustness | advanced | 51% | 50% | 48% |
| codex_0470 | theory_generalization | intermediate | 52% | 50% | 46% |
| codex_0485 | alignment_preference_learning | intermediate | 52% | 51% | 46% |
| codex_0230 | safety_robustness | advanced | 53% | 52% | 50% |
| codex_0137 | systems_deployment | intermediate | 53% | 53% | 47% |
| codex_0330 | systems_deployment | intermediate | 55% | 54% | 48% |
| codex_0038 | safety_robustness | advanced | 55% | 54% | 56% |
| codex_0417 | alignment_preference_learning | intermediate | 57% | 58% | 47% |
