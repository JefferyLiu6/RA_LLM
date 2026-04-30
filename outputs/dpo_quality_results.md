# DPO Quality Evaluation

References: `data/sft_test.jsonl`  
Cached generations: `outputs/dpo_before_after.md`  
Examples: 125

## Summary

The content metrics compare generated outputs against the held-out reference answer for the same prompt. `Content Alignment` is a heuristic weighted score: 60% reference-token F1, 25% key-term recall, and 15% supported-token rate.

| Model | Format Compliance | Content Alignment | Ref Token F1 | Key-Term Recall | Unsupported Terms | Follow-up Valid | Formulaic Rate | Grammar Flags | Avg Words |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| base | 84% | 30% | 28% | 24% | 55% | 100% | 0% | 0% | 77.8 |
| DPO | 100% | 83% | 84% | 81% | 13% | 100% | 100% | 2% | 123.6 |

## Interpretation

- `Format Compliance` is the strict template check: exact section headers, exactly three bullets, and a follow-up that ends with `?`.
- `Content Alignment` is not human judgment; it checks overlap with the held-out reference answer and whether generated terms are supported by the prompt/reference.
- `Formulaic Rate` is a warning signal for repeated learned phrasing. High format compliance with high formulaic rate usually means the next improvement should be DPO or style-diverse SFT data.

## DPO Failure / Risk Examples

| Prompt | Issue | Follow-up / Opening |
|--------|-------|---------------------|
| codex_0236 | plural_subject_uses, how_should_ranks | How should sparse MoE ranks tokens before sending them to experts? |
| codex_0122 | plural_subject_uses | Can self-critique detect subtle formatting mistakes? |
| codex_0244 | plural_subject_uses | How should adversarial examples be sampled for research notes? |

## Lowest DPO Content-Alignment Examples

| Prompt | Topic | Difficulty | Content Alignment | Ref Token F1 | Key-Term Recall |
|--------|-------|------------|:---:|:---:|:---:|
| codex_0417 | alignment_preference_learning | intermediate | 40% | 39% | 38% |
| codex_0271 | training_optimization | advanced | 48% | 49% | 42% |
| codex_0057 | theory_generalization | advanced | 49% | 49% | 43% |
| codex_0421 | architectures | intermediate | 51% | 50% | 46% |
| codex_0471 | training_optimization | advanced | 52% | 53% | 42% |
| codex_0371 | safety_robustness | advanced | 52% | 51% | 48% |
| codex_0383 | safety_robustness | advanced | 53% | 52% | 48% |
| codex_0137 | systems_deployment | intermediate | 54% | 53% | 47% |
| codex_0286 | evaluation_metrics | advanced | 54% | 53% | 49% |
| codex_0330 | systems_deployment | intermediate | 55% | 55% | 48% |
