# Rubric Quality Scores

Heuristic rubric applied to cached model outputs (no model reload needed). Each dimension is scored 0–3; total is out of 12.

| Model | Summary (0–3) | Bullets (0–3) | Limitation (0–3) | Follow-up (0–3) | **Total / 12** |
|-------|:---:|:---:|:---:|:---:|:---:|
| base | 3.00 | 1.55 | 2.91 | 3.00 | **10.45** |
| baseline | 2.00 | 1.45 | 2.00 | 2.73 | **8.18** |
| epochs_5 | 2.64 | 2.18 | 2.64 | 3.00 | **10.45** |
| lr_1e-4 | 1.64 | 1.45 | 2.00 | 2.91 | **8.00** |
| lr_5e-4 | 2.27 | 2.09 | 2.36 | 3.00 | **9.73** |
| qlora_4bit | 1.91 | 1.64 | 2.09 | 2.82 | **8.45** |
| rank_32 | 2.82 | 2.00 | 2.36 | 3.00 | **10.18** |
| rank_64 | 2.27 | 2.00 | 2.27 | 3.00 | **9.55** |
| rank_8 | 1.82 | 1.45 | 2.00 | 2.82 | **8.09** |

## Dimension guide

| Dimension | 0 | 1 | 2 | 3 |
|-----------|---|---|---|---|
| **Summary** | Missing / < 8 words | 8–14 words | ≥ 15 words | ≥ 15 words + high specificity |
| **Bullets** | Missing or < 5 w/bullet | 5–9 w/bullet | 10–14 w/bullet | ≥ 15 w/bullet |
| **Limitation** | Missing / 'None' | Present, < 8 words | 8–18 words | > 18 words + technical |
| **Follow-up** | Missing | Present, no '?' | Ends '?', < 8 words | Ends '?', ≥ 8 words |
