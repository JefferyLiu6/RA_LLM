# Content Quality Metrics

Averages across all test prompts.

| Model | Total Words | Lex Diversity | Specificity | Summary W | Avg Point W | Limitation W | Follow-up W | FU ends '?' |
|-------|-------------|---------------|-------------|----------|-------------|--------------|-------------|-------------|
| base | 103 | 0.751 | 0.751 | 99 | 9.6 | 37 | 16 | 100% |
| baseline | 78 | 0.834 | 0.802 | 14 | 9.9 | 13 | 11 | 100% |
| epochs_5 | 105 | 0.830 | 0.795 | 19 | 12.5 | 20 | 15 | 100% |
| lr_1e-4 | 74 | 0.806 | 0.750 | 14 | 9.6 | 13 | 12 | 100% |
| lr_5e-4 | 101 | 0.827 | 0.792 | 19 | 13.3 | 18 | 14 | 100% |
| qlora_4bit | 80 | 0.842 | 0.783 | 15 | 10.4 | 14 | 10 | 100% |
| rank_32 | 100 | 0.825 | 0.796 | 20 | 12.3 | 19 | 14 | 100% |
| rank_64 | 97 | 0.831 | 0.790 | 17 | 11.5 | 19 | 15 | 100% |
| rank_8 | 74 | 0.832 | 0.764 | 14 | 9.8 | 12 | 12 | 100% |

**Column guide**
- **Lex Diversity** — unique tokens / total tokens. Higher = more varied vocabulary.
- **Specificity** — content tokens / total tokens (excl. common filler words). Higher = denser information.
- **Summary W** — word count of the Summary sentence. More words often means more precise phrasing.
- **Avg Point W** — average words per Key Point bullet. Higher = more detailed explanations.
- **FU ends '?'** — fraction where the Follow-up Question is actually phrased as a question.
