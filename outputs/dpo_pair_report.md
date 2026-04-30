# DPO Pair Build Report

Output: `data/dpo_pairs.jsonl`
Pairs: 300
Input files:
- `data/sft_train.jsonl`
- `data/sft_val.jsonl`

## Split Counts

| Split | Count |
|-------|------:|
| train | 258 |
| val | 42 |

## Topic Counts

| Topic | Count |
|-------|------:|
| alignment_preference_learning | 45 |
| architectures | 39 |
| data_preprocessing | 36 |
| evaluation_metrics | 36 |
| safety_robustness | 36 |
| systems_deployment | 36 |
| theory_generalization | 36 |
| training_optimization | 36 |

## Difficulty Counts

| Difficulty | Count |
|------------|------:|
| advanced | 102 |
| intermediate | 99 |
| intro | 99 |

## Rejected-Issue Tags

| Tag | Count |
|-----|------:|
| content_drift | 100 |
| extra_bullet | 100 |
| extra_sentence_after_question | 100 |
| format_pass_negative | 100 |
| near_miss_reference_negative | 200 |
| unsupported_claim | 200 |

## Rejected Source Counts

| Source | Count |
|--------|------:|
| hard_negative_extra_bullet | 100 |
| hard_negative_extra_followup_sentence | 100 |
| hard_negative_unsupported_content | 100 |

## Validation

- Unique pair ids: 300/300
- Test-split pairs: 0
- Chosen strict-format pass: 300/300
- Rejected strict-format pass: 100/300

## Notes

- The held-out test split is not used by default.
- `chosen` responses use the validated train/val reference output by default.
- Default `rejected` responses are hard near-misses made from the same reference answer.
- The hard negatives target observed DPO failures: extra bullets, unsupported claims, and extra text after the follow-up question.
- Some rejected responses intentionally pass strict format so DPO also learns content quality rather than only schema compliance.
- Use `--negative-mode legacy` to reproduce the older formulaic-negative build.
- Pass `--chosen-mode rewrite` only for experimental local data generation; review those pairs carefully.
- Review a sample manually before using these pairs for a final DPO claim.
