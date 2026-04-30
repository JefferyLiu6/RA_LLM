PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
PORT ?= 8000
MERGED_MODEL ?= outputs/merged_model
MAC_OUTPUT_DIR ?= outputs/lora_adapter
CUDA_OUTPUT_DIR ?= outputs/cuda_lora_adapter
CUDA_QLORA_OUTPUT_DIR ?= outputs/cuda_qlora_adapter
SFT_TRAIN_PATH ?= data/sft_train.jsonl
SFT_VAL_PATH ?= data/sft_val.jsonl
SFT_TEST_PATH ?= data/sft_test.jsonl
EVAL_SMOKE_LIMIT ?= 10
SMOKE_OUTPUT_DIR ?= outputs/smoke_lora_adapter
DPO_OUTPUT ?= data/dpo_pairs.jsonl
DPO_REPORT ?= outputs/dpo_pair_report.md
DPO_MAX_PAIRS ?= 300
DPO_PAIRS_PER_RECORD ?= 3
DPO_NEGATIVE_MODE ?= hard
DPO_LEARNING_RATE ?= 1e-6
LOCAL_FILES_ONLY ?= 0
DPO_ADAPTER_DIR ?= outputs/dpo_adapter
DPO_SMOKE_OUTPUT_DIR ?= outputs/dpo_smoke_adapter
SFT_ADAPTER_DIR ?= $(MAC_OUTPUT_DIR)

.PHONY: help data-pipeline dataset-gold dataset-generate dataset-generate-openai dataset-build dataset-validate dataset-bootstrap dpo-pairs sft-quality dpo-quality readme-assets mac-check mac-train mac-train-sft mac-smoke-sft mac-smoke-dpo mac-train-dpo mac-sweep mac-eval mac-eval-sft-smoke mac-eval-sft mac-eval-dpo mac-content mac-rubric mac-dashboard mac-first mac-report cuda-check cuda-train cuda-train-sft cuda-smoke-sft cuda-smoke-dpo cuda-train-dpo cuda-qlora cuda-sweep cuda-eval cuda-eval-sft-smoke cuda-eval-sft cuda-eval-dpo cuda-vllm-serve

help:
	@printf "Dataset targets:\n"
	@printf "  make data-pipeline     Run ./data_pipeline.sh\n"
	@printf "  make dataset-bootstrap Convert current 68 examples, split, and validate\n"
	@printf "  make dataset-gold      Write data/dataset_gold.jsonl from data/dataset.jsonl\n"
	@printf "  make dataset-generate  Generate synthetic records locally, no API key\n"
	@printf "  make dataset-generate-openai Generate synthetic records with OpenAI API\n"
	@printf "  make dataset-build     Build data/sft_all/train/val/test JSONL files\n"
	@printf "  make dataset-validate  Validate data/sft_all.jsonl\n\n"
	@printf "DPO targets:\n"
	@printf "  make dpo-pairs         Build hard near-miss train/val DPO preference pairs\n\n"
	@printf "Evaluation targets:\n"
	@printf "  make sft-quality       Score cached SFT before/after outputs for content quality\n"
	@printf "  make dpo-quality       Score cached DPO before/after outputs for content quality\n\n"
	@printf "Presentation targets:\n"
	@printf "  make readme-assets     Regenerate README SVG result and pipeline visuals\n\n"
	@printf "Mac-first targets:\n"
	@printf "  make mac-check      Verify PyTorch MPS visibility\n"
	@printf "  make mac-train      Train standard LoRA on Mac/MPS\n"
	@printf "  make mac-train-sft  Train Mac/MPS LoRA on data/sft_train + data/sft_val\n"
	@printf "  make mac-smoke-sft  Two-step Mac/MPS smoke run on fixed SFT splits\n"
	@printf "  make mac-smoke-dpo  One-step Mac/MPS DPO smoke run\n"
	@printf "  make mac-train-dpo  Train DPO adapter on Mac/MPS\n"
	@printf "  make mac-sweep      Run Mac-safe LoRA sweep; QLoRA is skipped on MPS\n"
	@printf "  make mac-eval       Evaluate outputs/lora_adapter on test prompts\n"
	@printf "  make mac-eval-sft-smoke Evaluate first 10 examples from data/sft_test.jsonl\n"
	@printf "  make mac-eval-sft   Evaluate full data/sft_test.jsonl split\n"
	@printf "  make mac-eval-dpo   Evaluate outputs/dpo_adapter on data/sft_test.jsonl\n"
	@printf "  make mac-first      Run Mac check, train, and eval\n"
	@printf "  make mac-report     Run sweep-backed content metrics, rubric, and dashboard\n\n"
	@printf "CUDA targets:\n"
	@printf "  make cuda-check     Verify CUDA visibility\n"
	@printf "  make cuda-train     Train standard LoRA on CUDA with W&B enabled\n"
	@printf "  make cuda-train-sft Train CUDA LoRA on data/sft_train + data/sft_val\n"
	@printf "  make cuda-smoke-sft Two-step CUDA smoke run on fixed SFT splits\n"
	@printf "  make cuda-smoke-dpo One-step CUDA DPO smoke run\n"
	@printf "  make cuda-train-dpo Train DPO adapter on CUDA\n"
	@printf "  make cuda-qlora     Train 4-bit QLoRA on CUDA\n"
	@printf "  make cuda-sweep     Run full experiment sweep, including QLoRA\n"
	@printf "  make cuda-eval      Evaluate outputs/cuda_lora_adapter\n"
	@printf "  make cuda-eval-sft-smoke Evaluate first 10 examples from data/sft_test.jsonl\n"
	@printf "  make cuda-eval-sft  Evaluate full data/sft_test.jsonl split\n"
	@printf "  make cuda-eval-dpo  Evaluate outputs/dpo_adapter on data/sft_test.jsonl\n"
	@printf "  make cuda-vllm-serve Serve MERGED_MODEL with vLLM on CUDA\n"

data-pipeline:
	bash data_pipeline.sh

dataset-gold:
	$(PYTHON) src/dataset_pipeline.py gold

dataset-generate:
	$(PYTHON) src/dataset_pipeline.py generate-local

dataset-generate-openai:
	$(PYTHON) src/dataset_pipeline.py generate-openai

dataset-build:
	$(PYTHON) src/dataset_pipeline.py build

dataset-validate:
	$(PYTHON) src/dataset_pipeline.py validate

dataset-bootstrap: dataset-gold dataset-build dataset-validate

dpo-pairs:
	$(PYTHON) src/build_dpo_pairs.py --input $(SFT_TRAIN_PATH) --input $(SFT_VAL_PATH) --output $(DPO_OUTPUT) --report $(DPO_REPORT) --max-pairs $(DPO_MAX_PAIRS) --pairs-per-record $(DPO_PAIRS_PER_RECORD) --negative-mode $(DPO_NEGATIVE_MODE)

sft-quality:
	QUALITY_REPORT_TITLE="SFT Quality Evaluation" ADAPTER_LABEL=LoRA TEST_PROMPTS_PATH=$(SFT_TEST_PATH) BEFORE_AFTER_PATH=outputs/sft_before_after.md QUALITY_RESULTS_PATH=outputs/sft_quality_results.md QUALITY_DETAILS_PATH=outputs/sft_quality_details.csv $(PYTHON) src/eval_sft_quality.py

dpo-quality:
	QUALITY_REPORT_TITLE="DPO Quality Evaluation" ADAPTER_LABEL=DPO TEST_PROMPTS_PATH=$(SFT_TEST_PATH) BEFORE_AFTER_PATH=outputs/dpo_before_after.md QUALITY_RESULTS_PATH=outputs/dpo_quality_results.md QUALITY_DETAILS_PATH=outputs/dpo_quality_details.csv $(PYTHON) src/eval_sft_quality.py

readme-assets:
	$(PYTHON) src/render_readme_assets.py

mac-check:
	$(PYTHON) -c "import torch; assert torch.backends.mps.is_available(), 'MPS is not available; use a Mac Python/PyTorch build with MPS or run CUDA targets'; print('mps_available=True')"

mac-train:
	PYTORCH_ENABLE_MPS_FALLBACK=1 USE_QLORA=0 REPORT_TO=none RUN_NAME=mac-sft OUTPUT_DIR=$(MAC_OUTPUT_DIR) $(PYTHON) src/train_lora.py

mac-train-sft:
	PYTORCH_ENABLE_MPS_FALLBACK=1 USE_QLORA=0 REPORT_TO=none RUN_NAME=mac-sft-fixed TRAIN_PATH=$(SFT_TRAIN_PATH) VAL_PATH=$(SFT_VAL_PATH) OUTPUT_DIR=$(MAC_OUTPUT_DIR) $(PYTHON) src/train_lora.py

mac-smoke-sft:
	PYTORCH_ENABLE_MPS_FALLBACK=1 USE_QLORA=0 REPORT_TO=none RUN_NAME=mac-smoke-fixed TRAIN_PATH=$(SFT_TRAIN_PATH) VAL_PATH=$(SFT_VAL_PATH) OUTPUT_DIR=$(SMOKE_OUTPUT_DIR) MAX_STEPS=2 MAX_SEQ_LEN=256 BATCH_SIZE=1 GRAD_ACC=1 $(PYTHON) src/train_lora.py

mac-smoke-dpo:
	PYTORCH_ENABLE_MPS_FALLBACK=1 REQUIRE_DEVICE=mps REPORT_TO=none RUN_NAME=mac-dpo-smoke SFT_ADAPTER_DIR=$(SFT_ADAPTER_DIR) DPO_DATA_PATH=$(DPO_OUTPUT) OUTPUT_DIR=$(DPO_SMOKE_OUTPUT_DIR) MAX_STEPS=1 EPOCHS=1 EVAL_STRATEGY=no SAVE_STRATEGY=no MAX_SEQ_LEN=384 MAX_PROMPT_LEN=192 MAX_TARGET_LEN=192 BATCH_SIZE=1 GRAD_ACC=1 LEARNING_RATE=$(DPO_LEARNING_RATE) LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/train_dpo.py

mac-train-dpo:
	PYTORCH_ENABLE_MPS_FALLBACK=1 REQUIRE_DEVICE=mps REPORT_TO=none RUN_NAME=mac-dpo SFT_ADAPTER_DIR=$(SFT_ADAPTER_DIR) DPO_DATA_PATH=$(DPO_OUTPUT) OUTPUT_DIR=$(DPO_ADAPTER_DIR) EPOCHS=1 MAX_SEQ_LEN=512 MAX_PROMPT_LEN=256 MAX_TARGET_LEN=256 BATCH_SIZE=1 GRAD_ACC=4 LEARNING_RATE=$(DPO_LEARNING_RATE) LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/train_dpo.py

mac-sweep:
	PYTORCH_ENABLE_MPS_FALLBACK=1 REPORT_TO=none $(PYTHON) src/run_experiments.py --skip-done

mac-eval:
	ADAPTER_DIR=$(MAC_OUTPUT_DIR) BEFORE_AFTER_PATH=outputs/before_after.md EVAL_RESULTS_PATH=outputs/eval_results.md $(PYTHON) src/eval_template.py --regenerate

mac-eval-sft-smoke:
	PYTORCH_ENABLE_MPS_FALLBACK=1 ADAPTER_DIR=$(MAC_OUTPUT_DIR) TEST_PROMPTS_PATH=$(SFT_TEST_PATH) EVAL_LIMIT=$(EVAL_SMOKE_LIMIT) BEFORE_AFTER_PATH=outputs/sft_before_after_smoke.md EVAL_RESULTS_PATH=outputs/sft_eval_results_smoke.md LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/eval_template.py --regenerate

mac-eval-sft:
	PYTORCH_ENABLE_MPS_FALLBACK=1 ADAPTER_DIR=$(MAC_OUTPUT_DIR) TEST_PROMPTS_PATH=$(SFT_TEST_PATH) BEFORE_AFTER_PATH=outputs/sft_before_after.md EVAL_RESULTS_PATH=outputs/sft_eval_results.md LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/eval_template.py --regenerate

mac-eval-dpo:
	PYTORCH_ENABLE_MPS_FALLBACK=1 ADAPTER_LABEL=DPO ADAPTER_DIR=$(DPO_ADAPTER_DIR) TEST_PROMPTS_PATH=$(SFT_TEST_PATH) BEFORE_AFTER_PATH=outputs/dpo_before_after.md EVAL_RESULTS_PATH=outputs/dpo_eval_results.md LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/eval_template.py --regenerate

mac-content:
	$(PYTHON) src/eval_content.py --regenerate

mac-rubric:
	$(PYTHON) src/eval_rubric.py

mac-dashboard:
	$(PYTHON) src/dashboard.py --no-open

mac-first: mac-check mac-train mac-eval

mac-report: mac-sweep mac-content mac-rubric mac-dashboard

cuda-check:
	$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"

cuda-train:
	USE_QLORA=0 REPORT_TO=wandb RUN_NAME=cuda-sft OUTPUT_DIR=$(CUDA_OUTPUT_DIR) $(PYTHON) src/train_lora.py

cuda-train-sft:
	USE_QLORA=0 REPORT_TO=wandb RUN_NAME=cuda-sft-fixed TRAIN_PATH=$(SFT_TRAIN_PATH) VAL_PATH=$(SFT_VAL_PATH) OUTPUT_DIR=$(CUDA_OUTPUT_DIR) $(PYTHON) src/train_lora.py

cuda-smoke-sft:
	USE_QLORA=0 REPORT_TO=none RUN_NAME=cuda-smoke-fixed TRAIN_PATH=$(SFT_TRAIN_PATH) VAL_PATH=$(SFT_VAL_PATH) OUTPUT_DIR=$(SMOKE_OUTPUT_DIR) MAX_STEPS=2 MAX_SEQ_LEN=256 BATCH_SIZE=1 GRAD_ACC=1 $(PYTHON) src/train_lora.py

cuda-smoke-dpo:
	REQUIRE_DEVICE=cuda REPORT_TO=none RUN_NAME=cuda-dpo-smoke SFT_ADAPTER_DIR=$(SFT_ADAPTER_DIR) DPO_DATA_PATH=$(DPO_OUTPUT) OUTPUT_DIR=$(DPO_SMOKE_OUTPUT_DIR) MAX_STEPS=1 EPOCHS=1 EVAL_STRATEGY=no SAVE_STRATEGY=no MAX_SEQ_LEN=384 MAX_PROMPT_LEN=192 MAX_TARGET_LEN=192 BATCH_SIZE=1 GRAD_ACC=1 LEARNING_RATE=$(DPO_LEARNING_RATE) LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/train_dpo.py

cuda-train-dpo:
	REQUIRE_DEVICE=cuda REPORT_TO=wandb RUN_NAME=cuda-dpo SFT_ADAPTER_DIR=$(SFT_ADAPTER_DIR) DPO_DATA_PATH=$(DPO_OUTPUT) OUTPUT_DIR=$(DPO_ADAPTER_DIR) EPOCHS=1 MAX_SEQ_LEN=512 MAX_PROMPT_LEN=256 MAX_TARGET_LEN=256 BATCH_SIZE=2 GRAD_ACC=4 LEARNING_RATE=$(DPO_LEARNING_RATE) LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/train_dpo.py

cuda-qlora:
	USE_QLORA=1 BNB_BITS=4 REPORT_TO=wandb RUN_NAME=cuda-qlora OUTPUT_DIR=$(CUDA_QLORA_OUTPUT_DIR) $(PYTHON) src/train_lora.py

cuda-sweep:
	REPORT_TO=wandb $(PYTHON) src/run_experiments.py --skip-done

cuda-eval:
	ADAPTER_DIR=$(CUDA_OUTPUT_DIR) BEFORE_AFTER_PATH=outputs/cuda_before_after.md EVAL_RESULTS_PATH=outputs/cuda_eval_results.md $(PYTHON) src/eval_template.py --regenerate

cuda-eval-sft-smoke:
	ADAPTER_DIR=$(CUDA_OUTPUT_DIR) TEST_PROMPTS_PATH=$(SFT_TEST_PATH) EVAL_LIMIT=$(EVAL_SMOKE_LIMIT) BEFORE_AFTER_PATH=outputs/cuda_sft_before_after_smoke.md EVAL_RESULTS_PATH=outputs/cuda_sft_eval_results_smoke.md LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/eval_template.py --regenerate

cuda-eval-sft:
	ADAPTER_DIR=$(CUDA_OUTPUT_DIR) TEST_PROMPTS_PATH=$(SFT_TEST_PATH) BEFORE_AFTER_PATH=outputs/cuda_sft_before_after.md EVAL_RESULTS_PATH=outputs/cuda_sft_eval_results.md LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/eval_template.py --regenerate

cuda-eval-dpo:
	ADAPTER_LABEL=DPO ADAPTER_DIR=$(DPO_ADAPTER_DIR) TEST_PROMPTS_PATH=$(SFT_TEST_PATH) BEFORE_AFTER_PATH=outputs/cuda_dpo_before_after.md EVAL_RESULTS_PATH=outputs/cuda_dpo_eval_results.md LOCAL_FILES_ONLY=$(LOCAL_FILES_ONLY) $(PYTHON) src/eval_template.py --regenerate

cuda-vllm-serve:
	vllm serve $(MERGED_MODEL) --host 0.0.0.0 --port $(PORT)
