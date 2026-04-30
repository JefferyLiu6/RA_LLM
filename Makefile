PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
PORT ?= 8000
MERGED_MODEL ?= outputs/merged_model
MAC_OUTPUT_DIR ?= outputs/lora_adapter
CUDA_OUTPUT_DIR ?= outputs/cuda_lora_adapter
CUDA_QLORA_OUTPUT_DIR ?= outputs/cuda_qlora_adapter

.PHONY: help data-pipeline dataset-gold dataset-generate dataset-generate-openai dataset-build dataset-validate dataset-bootstrap mac-check mac-train mac-sweep mac-eval mac-content mac-rubric mac-dashboard mac-first mac-report cuda-check cuda-train cuda-qlora cuda-sweep cuda-eval cuda-vllm-serve

help:
	@printf "Dataset targets:\n"
	@printf "  make data-pipeline     Run ./data_pipeline.sh\n"
	@printf "  make dataset-bootstrap Convert current 68 examples, split, and validate\n"
	@printf "  make dataset-gold      Write data/dataset_gold.jsonl from data/dataset.jsonl\n"
	@printf "  make dataset-generate  Generate synthetic records locally, no API key\n"
	@printf "  make dataset-generate-openai Generate synthetic records with OpenAI API\n"
	@printf "  make dataset-build     Build data/sft_all/train/val/test JSONL files\n"
	@printf "  make dataset-validate  Validate data/sft_all.jsonl\n\n"
	@printf "Mac-first targets:\n"
	@printf "  make mac-check      Verify PyTorch MPS visibility\n"
	@printf "  make mac-train      Train standard LoRA on Mac/MPS\n"
	@printf "  make mac-sweep      Run Mac-safe LoRA sweep; QLoRA is skipped on MPS\n"
	@printf "  make mac-eval       Evaluate outputs/lora_adapter on test prompts\n"
	@printf "  make mac-first      Run Mac check, train, and eval\n"
	@printf "  make mac-report     Run sweep-backed content metrics, rubric, and dashboard\n\n"
	@printf "CUDA targets:\n"
	@printf "  make cuda-check     Verify CUDA visibility\n"
	@printf "  make cuda-train     Train standard LoRA on CUDA with W&B enabled\n"
	@printf "  make cuda-qlora     Train 4-bit QLoRA on CUDA\n"
	@printf "  make cuda-sweep     Run full experiment sweep, including QLoRA\n"
	@printf "  make cuda-eval      Evaluate outputs/cuda_lora_adapter\n"
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

mac-check:
	$(PYTHON) -c "import torch; assert torch.backends.mps.is_available(), 'MPS is not available; use a Mac Python/PyTorch build with MPS or run CUDA targets'; print('mps_available=True')"

mac-train:
	PYTORCH_ENABLE_MPS_FALLBACK=1 USE_QLORA=0 REPORT_TO=none RUN_NAME=mac-sft OUTPUT_DIR=$(MAC_OUTPUT_DIR) $(PYTHON) src/train_lora.py

mac-sweep:
	PYTORCH_ENABLE_MPS_FALLBACK=1 REPORT_TO=none $(PYTHON) src/run_experiments.py --skip-done

mac-eval:
	ADAPTER_DIR=$(MAC_OUTPUT_DIR) BEFORE_AFTER_PATH=outputs/before_after.md EVAL_RESULTS_PATH=outputs/eval_results.md $(PYTHON) src/eval_template.py --regenerate

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

cuda-qlora:
	USE_QLORA=1 BNB_BITS=4 REPORT_TO=wandb RUN_NAME=cuda-qlora OUTPUT_DIR=$(CUDA_QLORA_OUTPUT_DIR) $(PYTHON) src/train_lora.py

cuda-sweep:
	REPORT_TO=wandb $(PYTHON) src/run_experiments.py --skip-done

cuda-eval:
	ADAPTER_DIR=$(CUDA_OUTPUT_DIR) BEFORE_AFTER_PATH=outputs/cuda_before_after.md EVAL_RESULTS_PATH=outputs/cuda_eval_results.md $(PYTHON) src/eval_template.py --regenerate

cuda-vllm-serve:
	vllm serve $(MERGED_MODEL) --host 0.0.0.0 --port $(PORT)
