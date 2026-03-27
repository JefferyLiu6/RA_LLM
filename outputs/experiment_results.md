# LoRA Experiment Results

| Experiment | R | LR | Epochs | Train Loss | Val Loss | LoRA Compliance | Base Compliance |
|------------|---|-----|--------|------------|----------|-----------------|-----------------|
| baseline | 16 | 2e-04 | 3 | 1.7470 | 1.5891 | 9/10 (90%) | 1/10 (10%) |
| rank_8 | 8 | 2e-04 | 3 | 2.0143 | 1.8735 | 7/10 (70%) | 1/10 (10%) |
| rank_32 | 32 | 2e-04 | 3 | 1.4949 | 1.4506 | 9/10 (90%) | 1/10 (10%) |
| rank_64 | 64 | 2e-04 | 3 | 1.3309 | 1.4224 | 10/10 (100%) | 1/10 (10%) |
| epochs_5 | 16 | 2e-04 | 5 | 1.3594 | 1.4345 | 10/10 (100%) | 1/10 (10%) |
| lr_1e-4 | 16 | 1e-04 | 3 | 2.0573 | 1.9391 | 7/10 (70%) | 1/10 (10%) |
| lr_5e-4 | 16 | 5e-04 | 3 | 1.4132 | 1.4368 | 9/10 (90%) | 1/10 (10%) |

## Experiment Configurations

| Experiment | Hyperparameter Overrides |
|------------|--------------------------|
| baseline | all defaults |
| rank_8 | `LORA_R=8` |
| rank_32 | `LORA_R=32` |
| rank_64 | `LORA_R=64` |
| epochs_5 | `EPOCHS=5` |
| lr_1e-4 | `LEARNING_RATE=1e-4` |
| lr_5e-4 | `LEARNING_RATE=5e-4` |
