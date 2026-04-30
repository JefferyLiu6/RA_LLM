"""
Build and validate the SFT dataset.

The pipeline is intentionally Mac-friendly: metadata conversion, validation, and
splitting are local-only. Synthetic generation is optional and only runs when an
OpenAI API key and the openai package are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

TOPICS = [
    "architectures",
    "training_optimization",
    "evaluation_metrics",
    "alignment_preference_learning",
    "systems_deployment",
    "theory_generalization",
    "data_preprocessing",
    "safety_robustness",
]

DIFFICULTIES = ["intro", "intermediate", "advanced"]
SPLITS = ["train", "val", "test"]

TOPIC_HINTS = {
    "architectures": [
        "attention", "transformer", "cnn", "convolution", "rnn", "lstm",
        "bert", "gpt", "moe", "mixture", "gan", "vae", "diffusion",
        "resnet", "architecture", "adapter modules",
    ],
    "training_optimization": [
        "optimizer", "adam", "sgd", "learning rate", "gradient",
        "dropout", "normalization", "regularization", "loss", "clipping",
        "batch normalization", "curriculum", "fine-tuning",
    ],
    "evaluation_metrics": [
        "benchmark", "accuracy", "metric", "evaluation", "humaneval",
        "calibration", "perplexity", "f1", "auc", "bleu", "rouge",
    ],
    "alignment_preference_learning": [
        "rlhf", "dpo", "preference", "reward model", "ppo", "alignment",
        "human feedback", "instruction", "helpfulness", "harmlessness",
    ],
    "systems_deployment": [
        "quantization", "vllm", "latency", "throughput", "kv cache",
        "batching", "serving", "inference", "lora", "qlora", "awq",
        "memory", "gpu", "adapter",
    ],
    "theory_generalization": [
        "bias", "variance", "generalization", "risk", "erm", "theory",
        "optimal transport", "inductive", "capacity", "nash",
    ],
    "data_preprocessing": [
        "tokenization", "dataset", "preprocessing", "augmentation",
        "self-supervised", "masked", "contrastive", "labels",
    ],
    "safety_robustness": [
        "robust", "safety", "bias", "adversarial", "hallucination",
        "privacy", "fairness", "out-of-distribution", "uncertainty",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_input(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def input_sha256(text: str) -> str:
    return hashlib.sha256(normalize_input(text).encode("utf-8")).hexdigest()


def infer_topic(text: str) -> str:
    lowered = text.lower()
    scores = {
        topic: sum(1 for hint in hints if hint in lowered)
        for topic, hints in TOPIC_HINTS.items()
    }
    best, count = max(scores.items(), key=lambda item: item[1])
    return best if count else "architectures"


def infer_difficulty(text: str) -> str:
    words = len(text.split())
    advanced_terms = [
        "kl", "elbo", "nash", "differentiable", "autoregressive",
        "contrastive", "quantization", "preference", "calibration",
        "covariate", "reparameterization", "optimization",
    ]
    hits = sum(1 for term in advanced_terms if term in text.lower())
    if words > 95 or hits >= 3:
        return "advanced"
    if words > 45 or hits >= 1:
        return "intermediate"
    return "intro"


def format_output_is_compliant(output: str) -> tuple[bool, str]:
    sys.path.insert(0, str(ROOT / "src"))
    from eval_template import check_compliance  # noqa: PLC0415

    checks = check_compliance(output)
    if checks["all_pass"]:
        return True, ""
    failed = [key for key, value in checks.items() if value is False]
    return False, ",".join(failed)


def with_metadata(
    record: dict,
    *,
    record_id: str,
    source: str,
    split: str = "unassigned",
    generation_model: str | None = None,
) -> dict:
    input_text = record["input"].strip()
    output_text = record["output"].strip()
    enriched = {
        "id": record.get("id") or record_id,
        "input": input_text,
        "output": output_text,
        "source": record.get("source") or source,
        "topic": record.get("topic") or infer_topic(input_text),
        "difficulty": record.get("difficulty") or infer_difficulty(input_text),
        "split": record.get("split") or split,
        "input_sha256": input_sha256(input_text),
    }
    model = record.get("generation_model") or generation_model
    if model:
        enriched["generation_model"] = model
    return enriched


def convert_gold(args: argparse.Namespace) -> None:
    raw = read_jsonl(args.input)
    records = [
        with_metadata(record, record_id=f"gold_{i:04d}", source="human_gold")
        for i, record in enumerate(raw, 1)
    ]
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} gold records to {args.output}")


def generation_prompt(count: int, existing_topics: dict[str, int]) -> str:
    topic_lines = "\n".join(f"- {topic}: currently {n}" for topic, n in existing_topics.items())
    return f"""
Generate {count} supervised fine-tuning examples for a research-notes adapter.

Each example must take a technical ML concept, paragraph, or abstract and produce
the exact output template below.

Output template:
Summary:
<one-sentence summary>

Key Points:
- <point 1>
- <point 2>
- <point 3>

Limitation:
<one key limitation>

Follow-up Question:
<one question worth exploring>

Dataset constraints:
- Return diverse examples, not paraphrases of famous textbook definitions.
- Vary input length: short concept blurbs, medium technical paragraphs, and longer abstract-style inputs.
- Cover these topic buckets and prefer underrepresented buckets:
{topic_lines}
- Valid topic values: {", ".join(TOPICS)}
- Valid difficulty values: {", ".join(DIFFICULTIES)}
- Each output must contain exactly three Key Points bullets.
- Do not include markdown headers, numbered bullets, citations, or extra fields.
""".strip()


LOCAL_BLUEPRINTS = {
    "architectures": [
        ("Vision Transformer patch embeddings", "image patches are projected into token vectors before self-attention", "model global spatial relationships from the first layer", "it can require large datasets or strong augmentation to match CNN inductive bias", "How do hybrid CNN-ViT stems change data efficiency on small vision datasets?"),
        ("Rotary positional embeddings", "relative position information is encoded by rotating query and key vectors", "improve extrapolation compared with fixed absolute position tables", "long-context performance still depends on training distribution and scaling choices", "Can rotary scaling methods preserve short-context quality while extending maximum sequence length?"),
        ("Grouped-query attention", "multiple query heads share a smaller number of key-value heads", "reduce KV-cache memory and improve inference throughput", "too much sharing can reduce attention diversity and hurt generation quality", "What grouping ratio best balances memory savings against quality for small language models?"),
        ("State-space sequence models", "linear recurrent state updates replace full quadratic attention over tokens", "support long sequences with near-linear computational scaling", "they may struggle with retrieval tasks that require exact token-level lookup", "Can hybrid attention and state-space blocks outperform either architecture alone on long documents?"),
        ("Diffusion U-Net backbones", "a denoising network predicts noise at each timestep using skip connections", "generate high-quality images through iterative refinement", "sampling is slower than single-pass generative models without acceleration", "How much quality is lost when diffusion sampling is distilled into fewer denoising steps?"),
        ("Encoder-decoder transformers", "an encoder builds source representations and a decoder attends to them while generating", "fit translation, summarization, and structured generation tasks", "they are less convenient for pure continuation tasks than decoder-only models", "When does an encoder-decoder architecture beat decoder-only prompting for document summarization?"),
        ("Sparse MoE feed-forward layers", "a router sends each token to a small subset of expert networks", "increase parameter capacity without proportional FLOPs per token", "routing imbalance can underuse experts and create communication bottlenecks", "Do expert-choice routers stabilize MoE training better than token-choice routers?"),
        ("Retrieval-augmented generation", "a retriever supplies external passages that the generator conditions on", "ground outputs in up-to-date or private knowledge without retraining", "retrieval errors can inject irrelevant context that the generator over-trusts", "How should retriever confidence influence whether a model cites or ignores retrieved context?"),
    ],
    "training_optimization": [
        ("Cosine learning-rate decay", "the learning rate gradually decreases following a half-cosine schedule", "allows large early updates and smaller late-stage refinement", "poor warmup settings can still destabilize early training", "How sensitive are LoRA adapters to warmup ratio under cosine decay?"),
        ("Gradient accumulation", "several micro-batches are accumulated before one optimizer step", "simulate larger batch sizes when memory is limited", "optimizer updates become less frequent and training feedback is delayed", "When does gradient accumulation stop matching true large-batch training dynamics?"),
        ("Label smoothing", "hard one-hot targets are softened by assigning small probability to other classes", "reduce overconfidence and improve calibration", "too much smoothing can hide useful class distinctions", "Can label smoothing improve LLM reward-model calibration without weakening ranking accuracy?"),
        ("Weight decay in AdamW", "parameter decay is decoupled from adaptive gradient updates", "regularize weights without interacting incorrectly with Adam moments", "the best decay value varies sharply by architecture and dataset size", "Should adapter weights use lower decay than full fine-tuning weights?"),
        ("Gradient checkpointing", "intermediate activations are recomputed during backpropagation instead of stored", "reduce memory usage and enable larger batches or longer sequences", "training becomes slower because forward computations are repeated", "Where should checkpoint boundaries be placed to maximize memory savings per extra FLOP?"),
        ("Early stopping", "training stops when validation metrics stop improving", "reduce overfitting and wasted compute", "noisy validation curves can stop a run before it reaches a better basin", "How many validation checks are needed before early stopping is reliable for small datasets?"),
        ("Curriculum fine-tuning", "examples are ordered from easier to harder during training", "help the model learn simple structure before harder variations", "a bad curriculum can bias the model away from difficult examples", "Can curriculum ordering improve template adherence without reducing content quality?"),
        ("Low-rank adapter dropout", "dropout is applied inside the LoRA branch during training", "regularize adapter updates on small datasets", "excessive dropout can underfit the target format", "What dropout rate gives the best compliance-quality tradeoff for small LoRA datasets?"),
    ],
    "evaluation_metrics": [
        ("Expected calibration error", "confidence scores are bucketed and compared with empirical accuracy", "measure whether predicted probabilities match observed correctness", "bucket choices can make calibration estimates unstable on small test sets", "How should calibration be reported when the evaluation set has only a few hundred examples?"),
        ("Pairwise win rate", "a judge compares two model outputs for the same prompt", "capture relative preference even when absolute scoring is hard", "judge bias can favor verbosity or familiar phrasing over correctness", "How can human audits estimate whether an LLM judge is systematically biased?"),
        ("Exact format compliance", "outputs are parsed for required headers, sections, and bullet counts", "give a deterministic metric for structured-output reliability", "format success does not prove that the content is factually useful", "Which content-quality metric should be paired with compliance for structured notes?"),
        ("BERTScore", "contextual token embeddings are matched between candidate and reference text", "reward semantic similarity beyond exact n-gram overlap", "high embedding similarity can miss factual contradictions", "Can BERTScore detect when a limitation is plausible but wrong?"),
        ("Pass@k", "multiple generated candidates are checked and success is counted if any pass", "estimate functional correctness for stochastic code generation", "large k can hide poor first-sample reliability", "How should pass@k be reported alongside latency and sample budget?"),
        ("Inter-annotator agreement", "multiple human labels are compared for consistency", "quantify whether the rubric is clear enough for reliable evaluation", "low agreement can reflect either ambiguity or insufficient annotator training", "What agreement threshold is acceptable before using human labels as ground truth?"),
        ("Ablation tables", "one experimental variable is changed while others are fixed", "identify which training choices actually drive improvements", "interactions between variables can be missed by one-factor sweeps", "When should a rank sweep be replaced by a factorial experiment?"),
        ("Latency percentiles", "p50 and p95 latencies summarize typical and tail response times", "show whether serving is reliable under realistic request variation", "single-user benchmarks can underestimate production queueing effects", "How should latency be benchmarked for batched LLM serving on one GPU?"),
    ],
    "alignment_preference_learning": [
        ("Direct Preference Optimization", "chosen and rejected responses define a classification-style preference objective", "avoid training a separate reward model and running online RL", "preference pairs must be high quality or the model learns judge artifacts", "How much manual audit is needed before synthetic DPO pairs are trustworthy?"),
        ("RLHF reward modeling", "a model is trained to predict human preference rankings", "turn qualitative feedback into a scalar optimization signal", "reward models can be exploited by policies that find shortcut behaviors", "Can reward-model uncertainty be used to reject risky policy updates?"),
        ("Constitutional AI", "model critiques and revisions are guided by written principles", "scale preference data creation with less direct human labeling", "principles may be incomplete or conflict in edge cases", "How should constitutional rules be tested on domain-specific technical writing tasks?"),
        ("Rejection sampling fine-tuning", "several outputs are generated and only high-scoring samples are used for SFT", "improve data quality without changing the training objective", "the generator and judge can reinforce each other's biases", "When does rejection sampling outperform DPO for structured generation?"),
        ("KL regularization", "policy updates are penalized for moving too far from a reference model", "preserve base capabilities while improving aligned behavior", "too strong a penalty prevents useful adaptation", "How should KL strength be tuned for small adapters rather than full models?"),
        ("Preference data margins", "pairs are kept only when a judge strongly prefers one response", "reduce noisy labels in DPO training", "strict margins can discard useful but subtle preferences", "What margin threshold best predicts human agreement on preference pairs?"),
        ("Helpful-harmless tradeoff", "models are optimized for utility while avoiding unsafe or misleading answers", "make alignment evaluation multidimensional", "overemphasis on refusal can reduce helpfulness on benign prompts", "How should domain-specific assistants balance caution and useful technical detail?"),
        ("Self-critique loops", "a model evaluates and revises its own draft before finalizing", "improve structure and catch obvious omissions", "the model may fail to notice errors it already made", "Can self-critique improve small-model outputs after LoRA fine-tuning?"),
    ],
    "systems_deployment": [
        ("KV-cache reuse", "past key and value tensors are stored during autoregressive decoding", "avoid recomputing attention over the full prefix each token", "cache memory grows with sequence length and batch size", "How should KV-cache limits be set for serving long research abstracts?"),
        ("Continuous batching", "incoming requests are dynamically batched at each decoding step", "increase GPU utilization for variable-length generation workloads", "batching can raise tail latency for short interactive requests", "What scheduling policy gives the best throughput-latency tradeoff in vLLM?"),
        ("AWQ weight quantization", "important weights are preserved while less sensitive weights are quantized", "reduce model size with smaller quality loss than naive int4 quantization", "calibration data must match expected inference prompts", "How sensitive is AWQ quality to calibration examples for structured notes?"),
        ("Paged attention", "KV-cache blocks are managed like virtual memory pages", "serve many requests with less fragmentation", "implementation complexity can make performance hardware-dependent", "How does paged attention change throughput for mixed prompt lengths?"),
        ("LoRA adapter merging", "low-rank updates are folded into the base weights for inference", "remove adapter overhead and simplify serving", "merged weights lose the ability to swap adapters dynamically", "When should production keep adapters separate instead of merging them?"),
        ("Tensor parallel inference", "model weights are split across multiple GPUs", "serve models too large for one device and improve throughput", "communication overhead can dominate for small models", "Is tensor parallelism worthwhile for 1.5B models on consumer GPUs?"),
        ("Speculative decoding", "a small draft model proposes tokens that a larger model verifies", "reduce latency when draft predictions are accepted", "speedup disappears if draft and target distributions diverge", "Can a fine-tuned small model be a useful draft model for its base model?"),
        ("Throughput benchmarking", "tokens per second are measured under fixed prompt and generation settings", "compare serving configurations with a single operational metric", "numbers are easy to inflate with unrealistic batch sizes", "What benchmark mix best represents an interactive research assistant workload?"),
    ],
    "theory_generalization": [
        ("Bias-variance decomposition", "prediction error is separated into systematic bias, variance, and noise", "explain underfitting and overfitting tradeoffs", "the classical decomposition is less direct for modern overparameterized models", "How does double descent complicate the traditional bias-variance story?"),
        ("Double descent", "test error can rise and then fall again as model capacity increases", "explain why highly overparameterized models can generalize well", "the exact interpolation threshold depends on data and optimization details", "Can adapter rank sweeps show a small-scale version of double descent?"),
        ("Minimum description length", "models are preferred when they compress data with shorter descriptions", "connect generalization to simplicity and compression", "practical neural network descriptions are hard to define precisely", "Can compression-based metrics predict which fine-tuned adapter will generalize best?"),
        ("PAC-Bayes bounds", "generalization is bounded using a prior and posterior over predictors", "give probabilistic guarantees for randomized learned models", "bounds are often too loose to explain practical deep learning performance", "Can PAC-Bayes analysis meaningfully compare LoRA ranks?"),
        ("Information bottleneck", "representations are encouraged to preserve task-relevant information while discarding noise", "frame learning as compression of inputs into useful features", "estimating mutual information in high-dimensional networks is difficult", "Do adapter layers create measurable bottlenecks in transformer representations?"),
        ("Sharpness-aware minimization", "optimization prefers parameters whose neighborhoods have low loss", "seek flatter minima associated with better generalization", "sharpness depends on parameter scaling and is not invariant", "Can SAM improve small-data LoRA training without destabilizing adapters?"),
        ("Out-of-distribution generalization", "models are tested on data shifted from the training distribution", "measure robustness beyond memorized training patterns", "defining realistic shifts is task-specific and easy to oversimplify", "Which topic shifts should be held out for research-note adapters?"),
        ("No-free-lunch theorem", "no learner performs best across all possible data-generating distributions", "explain why inductive bias is necessary for learning", "the theorem is too broad to guide practical model selection alone", "How should inductive bias be stated when comparing LoRA and full fine-tuning?"),
    ],
    "data_preprocessing": [
        ("Byte-level tokenization", "raw bytes are modeled directly rather than segmented into subwords", "handle arbitrary languages, code, and noisy text uniformly", "sequence lengths become much longer than subword tokenization", "When does tokenizer-free modeling justify the extra sequence length?"),
        ("Subword vocabulary training", "frequent character sequences are merged into reusable tokens", "balance compact sequence length with open-vocabulary coverage", "rare words can still be split into unintuitive fragments", "How does vocabulary choice affect technical acronym handling?"),
        ("Data deduplication", "near-identical examples are removed before training or evaluation", "reduce memorization and prevent train-test leakage", "aggressive deduplication can remove legitimate recurring patterns", "What similarity threshold catches leakage without deleting useful template examples?"),
        ("Length bucketing", "examples of similar sequence length are batched together", "reduce padding waste and improve training efficiency", "buckets can interact with curriculum effects if not shuffled well", "Does length bucketing change convergence for short structured-output datasets?"),
        ("Data augmentation", "inputs are transformed while preserving the desired label", "increase coverage and robustness under limited data", "bad augmentations can change meaning and introduce label noise", "Which augmentations preserve correctness for technical ML explanations?"),
        ("Train-validation-test splits", "data is partitioned into model-fitting, model-selection, and final-evaluation sets", "separate tuning decisions from final claims", "small datasets can make split estimates noisy", "How large should the held-out test set be for a narrow formatting task?"),
        ("Weak supervision", "labels are produced by heuristics or teacher models", "scale datasets when manual labeling is expensive", "systematic labeler errors can become training artifacts", "How should weakly supervised examples be audited before fine-tuning?"),
        ("Prompt normalization", "inputs are cleaned into consistent whitespace and formatting", "make hashing, deduplication, and parsing reliable", "over-normalization can erase meaningful structure such as code indentation", "Which normalization rules are safe for mixed prose and code prompts?"),
    ],
    "safety_robustness": [
        ("Adversarial prompts", "inputs are crafted to trigger failures or policy violations", "stress-test behavior beyond average-case examples", "manual adversarial sets can overfit to known attack styles", "How should adversarial prompts be refreshed after each model update?"),
        ("Hallucination detection", "outputs are checked for unsupported or fabricated claims", "reduce the risk of fluent but false technical explanations", "automatic detectors often confuse uncertainty with hallucination", "Can citation-free technical summaries be judged for factual support reliably?"),
        ("Robustness to long inputs", "models are evaluated on contexts longer than typical training examples", "expose truncation and attention failure modes", "long tests are expensive and may conflate comprehension with memory limits", "How should long-context robustness be measured for structured notes?"),
        ("Prompt injection", "malicious instructions inside user or retrieved text try to override system goals", "test whether the model follows the intended instruction hierarchy", "some attacks are hard to distinguish from legitimate task content", "Can fine-tuning improve resistance to prompt injection without hurting helpfulness?"),
        ("Distribution shift", "test data differs from training data in topic, style, or difficulty", "measure whether a model learned robust behavior rather than memorized patterns", "choosing shifts after seeing failures can bias the evaluation", "Which held-out topics best reveal overfitting in an ML-notes adapter?"),
        ("Uncertainty-aware refusal", "a model signals when it cannot answer confidently", "avoid overclaiming on ambiguous or under-specified prompts", "excessive uncertainty can make a benign assistant less useful", "How should uncertainty be calibrated for educational technical summaries?"),
        ("Sensitive data memorization", "models reproduce private or rare training strings", "raise privacy risks when training data includes confidential content", "small adapters can still memorize examples even when the base is frozen", "How can canary strings estimate memorization risk in LoRA fine-tuning?"),
        ("Factual consistency checks", "generated summaries are compared against source input claims", "catch contradictions introduced during compression", "semantic consistency is difficult to verify with simple string rules", "Can LLM-as-judge consistency checks be validated against human annotations?"),
    ],
}


INPUT_PATTERNS = [
    "{concept} is often introduced as a practical answer to a specific ML bottleneck: {mechanism}. Its value is that it can {benefit}, though a careful implementation still has to handle cases where {limitation}.",
    "{concept} becomes useful when a team needs to {benefit}. The method depends on the detail that {mechanism}, and that detail can become fragile when {limitation}.",
    "{concept} can be evaluated by asking whether the mechanism, where {mechanism}, actually supports the intended benefit: {benefit}. A negative result would often show up when {limitation}.",
    "{concept} creates a trade-off between capability and reliability. It can {benefit} because {mechanism}, but the same setup is less convincing when {limitation}.",
    "{concept} appears in production systems when engineers want to {benefit}. The implementation usually relies on the fact that {mechanism}; the operational risk is that {limitation}.",
    "{concept} is a useful case study for adapter-style research notes. The key technical idea is that {mechanism}. This matters because it can {benefit}, but it should not be treated as solved if {limitation}.",
    "{concept} is usually discussed as more than a formatting trick. In a realistic system, {mechanism}, which can {benefit}. The hard part is verifying behavior when {limitation}.",
    "{concept} matters in research pipelines because it offers a way to {benefit}. The underlying mechanism is that {mechanism}. Any benchmark should include examples where {limitation}.",
    "{concept} should be stress-tested under conditions that expose its main weakness. The method can {benefit} by relying on the mechanism that {mechanism}, but failures are likely when {limitation}.",
]


def local_output(
    concept: str,
    mechanism: str,
    benefit: str,
    limitation: str,
    question: str,
    variant: int,
) -> str:
    limitation_sentence = limitation[:1].upper() + limitation[1:]
    if variant == 0:
        summary = (
            f"{concept} uses a targeted modeling or systems mechanism to {benefit}, "
            f"while requiring care because {limitation}."
        )
        points = [
            f"Its core mechanism is that {mechanism}.",
            f"The main practical benefit is that it can {benefit}.",
            f"A useful evaluation should include cases where {limitation}.",
        ]
        limitation_text = (
            f"{limitation_sentence}, so results should be validated on realistic held-out examples "
            "rather than inferred from the method name alone."
        )
    elif variant == 1:
        summary = (
            f"{concept} is useful because it can {benefit}, but its reliability depends on "
            f"how well the underlying mechanism handles cases where {limitation}."
        )
        points = [
            f"The method operates through a setup where {mechanism}.",
            f"It is most valuable in workflows that need to {benefit}.",
            f"The most important failure mode appears when {limitation}.",
        ]
        limitation_text = (
            f"{limitation_sentence}, which means benchmark results can look stronger than real deployment behavior."
        )
    else:
        summary = (
            f"{concept} connects a concrete technical mechanism with a practical goal: "
            f"{mechanism}, enabling systems to {benefit}."
        )
        points = [
            f"The method should be explained in terms of its mechanism: {mechanism}.",
            f"Its expected upside is that teams can {benefit}.",
            f"A strong evaluation should include stress cases where {limitation}.",
        ]
        limitation_text = (
            f"{limitation_sentence}; without targeted tests, the method may appear reliable only because the evaluation is too narrow."
        )

    return (
        "Summary:\n"
        f"{summary}\n\n"
        "Key Points:\n"
        f"- {points[0]}\n"
        f"- {points[1]}\n"
        f"- {points[2]}\n\n"
        "Limitation:\n"
        f"{limitation_text}\n\n"
        "Follow-up Question:\n"
        f"{question}"
    )


def generate_local(args: argparse.Namespace) -> None:
    existing = read_jsonl(args.output) if args.append else []
    if len(existing) >= args.count:
        print(f"{args.output} already has {len(existing)} records; nothing to generate.")
        return

    records = list(existing)
    seen_hashes = {r.get("input_sha256") for r in records}
    next_idx = len(records) + 1

    generated: list[dict] = []
    for topic in TOPICS:
        for concept, mechanism, benefit, limitation, question in LOCAL_BLUEPRINTS[topic]:
            for pattern_index, pattern in enumerate(INPUT_PATTERNS):
                difficulty = DIFFICULTIES[pattern_index % len(DIFFICULTIES)]
                input_text = pattern.format(
                    concept=concept,
                    mechanism=mechanism,
                    benefit=benefit,
                    limitation=limitation,
                )
                record = {
                    "input": input_text,
                    "output": local_output(
                        concept,
                        mechanism,
                        benefit,
                        limitation,
                        question,
                        variant=pattern_index % 3,
                    ),
                    "topic": topic,
                    "difficulty": difficulty,
                }
                generated.append(record)

    random.Random(args.seed).shuffle(generated)

    for base in generated:
        if len(records) >= args.count:
            break
        enriched = with_metadata(
            base,
            record_id=f"codex_{next_idx:04d}",
            source="synthetic_codex",
            generation_model=args.model,
        )
        if enriched["input_sha256"] in seen_hashes:
            continue
        ok, reason = format_output_is_compliant(enriched["output"])
        if not ok:
            print(f"Skipping non-compliant local record: {reason}")
            continue
        records.append(enriched)
        seen_hashes.add(enriched["input_sha256"])
        next_idx += 1

    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} local synthetic records to {args.output}")


def generate_openai(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise SystemExit(
            "OpenAI generation requires optional dependencies. Run `pip install openai pydantic`."
        ) from exc

    class GeneratedRecord(BaseModel):
        input: str = Field(description="Technical ML input paragraph.")
        output: str = Field(description="Structured research notes in the required template.")
        topic: Literal[
            "architectures",
            "training_optimization",
            "evaluation_metrics",
            "alignment_preference_learning",
            "systems_deployment",
            "theory_generalization",
            "data_preprocessing",
            "safety_robustness",
        ]
        difficulty: Literal["intro", "intermediate", "advanced"]

    class GeneratedBatch(BaseModel):
        records: list[GeneratedRecord]

    existing = read_jsonl(args.output) if args.append else []
    if len(existing) >= args.count:
        print(f"{args.output} already has {len(existing)} records; nothing to generate.")
        return

    client = OpenAI()
    records = list(existing)
    next_idx = len(records) + 1

    while len(records) < args.count:
        batch_count = min(args.batch_size, args.count - len(records))
        topic_counts = Counter(r.get("topic", "unknown") for r in records)
        prompt = generation_prompt(batch_count, dict(topic_counts))

        response = client.responses.parse(
            model=args.model,
            input=[
                {
                    "role": "system",
                    "content": "You generate high-quality JSON dataset records for ML fine-tuning.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=GeneratedBatch,
            temperature=args.temperature,
        )

        parsed_batch = None
        for output in response.output:
            if output.type != "message":
                continue
            for item in output.content:
                if getattr(item, "type", None) == "refusal":
                    raise RuntimeError(f"Model refused generation: {item.refusal}")
                parsed = getattr(item, "parsed", None)
                if parsed:
                    parsed_batch = parsed
                    break
            if parsed_batch:
                break

        if not parsed_batch:
            raise RuntimeError("OpenAI response did not include parsed records.")

        accepted = 0
        for generated in parsed_batch.records:
            base = generated.model_dump()
            ok, reason = format_output_is_compliant(base["output"])
            if not ok:
                print(f"Skipping non-compliant generated record: {reason}")
                continue
            enriched = with_metadata(
                base,
                record_id=f"syn_{next_idx:04d}",
                source="synthetic_openai",
                generation_model=args.model,
            )
            if enriched["input_sha256"] in {r["input_sha256"] for r in records}:
                print(f"Skipping duplicate generated input: {enriched['id']}")
                continue
            records.append(enriched)
            next_idx += 1
            accepted += 1

        write_jsonl(args.output, records)
        print(f"Generated batch: accepted {accepted}; total {len(records)}/{args.count}")


def desired_split_counts(total: int, train: int, val: int, test: int) -> tuple[int, int, int]:
    requested = train + val + test
    if total >= requested:
        return train, val, test
    train_ratio = train / requested
    val_ratio = val / requested
    train_n = max(1, round(total * train_ratio)) if total else 0
    val_n = max(1, round(total * val_ratio)) if total >= 3 else 0
    test_n = total - train_n - val_n
    if total >= 3 and test_n < 1:
        train_n -= 1
        test_n = 1
    return train_n, val_n, test_n


def dedupe_records(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for record in records:
        digest = record.get("input_sha256") or input_sha256(record["input"])
        if digest in seen:
            continue
        seen.add(digest)
        unique.append({**record, "input_sha256": digest})
    return unique


def seeded_shuffle(records: list[dict], seed: int) -> list[dict]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def split_records(
    records: list[dict],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    records = seeded_shuffle(records, seed)
    train_n, val_n, test_n = desired_split_counts(len(records), train_count, val_count, test_count)

    non_gold = [r for r in records if r.get("source") != "human_gold"]
    test_pool = non_gold if len(non_gold) >= test_n else records
    test_ids = {r["id"] for r in seeded_shuffle(test_pool, seed + 1)[:test_n]}
    test = [r for r in records if r["id"] in test_ids]
    remaining = [r for r in records if r["id"] not in test_ids]

    val_ids = {r["id"] for r in seeded_shuffle(remaining, seed + 2)[:val_n]}
    val = [r for r in remaining if r["id"] in val_ids]
    train = [r for r in remaining if r["id"] not in val_ids][:train_n]
    return train, val, test


def set_split(records: list[dict], split: str) -> list[dict]:
    return [{**record, "split": split, "input_sha256": input_sha256(record["input"])} for record in records]


def build_splits(args: argparse.Namespace) -> None:
    gold = read_jsonl(args.gold)
    synthetic = []
    for path in args.synthetic:
        synthetic.extend(read_jsonl(path))

    if not gold:
        raise SystemExit(f"No gold records found at {args.gold}. Run `make dataset-gold` first.")

    normalized = []
    for i, record in enumerate(gold, 1):
        normalized.append(with_metadata(record, record_id=f"gold_{i:04d}", source="human_gold"))
    for i, record in enumerate(synthetic, 1):
        normalized.append(with_metadata(record, record_id=f"syn_{i:04d}", source="synthetic_openai"))

    unique = dedupe_records(normalized)
    train, val, test = split_records(
        unique,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )

    train = set_split(train, "train")
    val = set_split(val, "val")
    test = set_split(test, "test")
    all_records = train + val + test

    write_jsonl(args.all_output, all_records)
    write_jsonl(args.train_output, train)
    write_jsonl(args.val_output, val)
    write_jsonl(args.test_output, test)

    print(f"Wrote {len(all_records)} records to {args.all_output}")
    print(f"Split counts: train={len(train)} val={len(val)} test={len(test)}")


def validate_records(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    required = {"id", "input", "output", "source", "topic", "difficulty", "split", "input_sha256"}
    errors: list[str] = []
    ids: set[str] = set()
    hashes: set[str] = set()

    split_counts = Counter()
    source_counts = Counter()
    topic_counts = Counter()
    difficulty_counts = Counter()
    compliant = 0

    for line_no, record in enumerate(records, 1):
        missing = required - set(record)
        if missing:
            errors.append(f"line {line_no}: missing {sorted(missing)}")
            continue

        if record["id"] in ids:
            errors.append(f"line {line_no}: duplicate id {record['id']}")
        ids.add(record["id"])

        expected_hash = input_sha256(record["input"])
        if record["input_sha256"] != expected_hash:
            errors.append(f"line {line_no}: input_sha256 does not match normalized input")

        if record["input_sha256"] in hashes:
            errors.append(f"line {line_no}: duplicate input_sha256 {record['input_sha256']}")
        hashes.add(record["input_sha256"])

        if record["topic"] not in TOPICS:
            errors.append(f"line {line_no}: invalid topic {record['topic']}")
        if record["difficulty"] not in DIFFICULTIES:
            errors.append(f"line {line_no}: invalid difficulty {record['difficulty']}")
        if record["split"] not in SPLITS:
            errors.append(f"line {line_no}: invalid split {record['split']}")

        ok, reason = format_output_is_compliant(record["output"])
        if ok:
            compliant += 1
        else:
            errors.append(f"line {line_no}: output non-compliant ({reason})")

        split_counts[record["split"]] += 1
        source_counts[record["source"]] += 1
        topic_counts[record["topic"]] += 1
        difficulty_counts[record["difficulty"]] += 1

    report = render_report(
        records=records,
        errors=errors,
        split_counts=split_counts,
        source_counts=source_counts,
        topic_counts=topic_counts,
        difficulty_counts=difficulty_counts,
        compliant=compliant,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)
    print(f"Dataset report written to: {args.report}")

    if errors:
        raise SystemExit(1)


def render_counter_table(title: str, counter: Counter) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "|-------|------:|"]
    for key, value in sorted(counter.items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return lines


def input_diversity_stats(records: list[dict]) -> dict:
    if not records:
        return {
            "avg_words": 0,
            "unique_opening_4": 0,
            "max_opening_4_count": 0,
            "top_openings": [],
        }
    opening_counts = Counter(" ".join(record["input"].split()[:4]) for record in records)
    avg_words = sum(len(record["input"].split()) for record in records) / len(records)
    return {
        "avg_words": round(avg_words, 1),
        "unique_opening_4": len(opening_counts),
        "max_opening_4_count": max(opening_counts.values()),
        "top_openings": opening_counts.most_common(8),
    }


def render_report(
    *,
    records: list[dict],
    errors: list[str],
    split_counts: Counter,
    source_counts: Counter,
    topic_counts: Counter,
    difficulty_counts: Counter,
    compliant: int,
) -> str:
    lines = [
        "# Dataset Validation Report",
        "",
        f"Total records: {len(records)}",
        f"Template-compliant outputs: {compliant}/{len(records)}",
        f"Status: {'PASS' if not errors else 'FAIL'}",
        "",
    ]
    lines += render_counter_table("Splits", split_counts)
    lines += render_counter_table("Sources", source_counts)
    lines += render_counter_table("Topics", topic_counts)
    lines += render_counter_table("Difficulty", difficulty_counts)

    diversity = input_diversity_stats(records)
    lines += [
        "## Input Diversity",
        "",
        f"Average input words: {diversity['avg_words']}",
        f"Unique 4-word openings: {diversity['unique_opening_4']}",
        f"Most repeated 4-word opening count: {diversity['max_opening_4_count']}",
        "",
        "| Opening | Count |",
        "|---------|------:|",
    ]
    for opening, count in diversity["top_openings"]:
        lines.append(f"| {opening} | {count} |")
    lines.append("")

    lines += ["## Errors", ""]
    if errors:
        lines.extend(f"- {error}" for error in errors[:100])
        if len(errors) > 100:
            lines.append(f"- ... {len(errors) - 100} more errors")
    else:
        lines.append("No validation errors.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset generation, splitting, and validation pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    gold = sub.add_parser("gold", help="Convert data/dataset.jsonl into metadata-rich gold records.")
    gold.add_argument("--input", type=Path, default=DATA_DIR / "dataset.jsonl")
    gold.add_argument("--output", type=Path, default=DATA_DIR / "dataset_gold.jsonl")
    gold.set_defaults(func=convert_gold)

    gen = sub.add_parser("generate-openai", help="Generate synthetic records with OpenAI Structured Outputs.")
    gen.add_argument("--output", type=Path, default=DATA_DIR / "synthetic_openai.jsonl")
    gen.add_argument("--count", type=int, default=532)
    gen.add_argument("--batch-size", type=int, default=25)
    gen.add_argument("--model", default=os.getenv("GEN_MODEL", "gpt-4o-mini"))
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    gen.set_defaults(func=generate_openai)

    local = sub.add_parser("generate-local", help="Generate deterministic synthetic records without an API key.")
    local.add_argument("--output", type=Path, default=DATA_DIR / "synthetic_codex.jsonl")
    local.add_argument("--count", type=int, default=532)
    local.add_argument("--model", default="codex_session")
    local.add_argument("--seed", type=int, default=123)
    local.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    local.set_defaults(func=generate_local)

    build = sub.add_parser("build", help="Build sft_all/train/val/test JSONL files.")
    build.add_argument("--gold", type=Path, default=DATA_DIR / "dataset_gold.jsonl")
    build.add_argument(
        "--synthetic",
        type=Path,
        nargs="*",
        default=[DATA_DIR / "synthetic_codex.jsonl", DATA_DIR / "synthetic_openai.jsonl"],
    )
    build.add_argument("--all-output", type=Path, default=DATA_DIR / "sft_all.jsonl")
    build.add_argument("--train-output", type=Path, default=DATA_DIR / "sft_train.jsonl")
    build.add_argument("--val-output", type=Path, default=DATA_DIR / "sft_val.jsonl")
    build.add_argument("--test-output", type=Path, default=DATA_DIR / "sft_test.jsonl")
    build.add_argument("--train-count", type=int, default=400)
    build.add_argument("--val-count", type=int, default=75)
    build.add_argument("--test-count", type=int, default=125)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(func=build_splits)

    validate = sub.add_parser("validate", help="Validate a metadata-rich dataset JSONL.")
    validate.add_argument("--input", type=Path, default=DATA_DIR / "sft_all.jsonl")
    validate.add_argument("--report", type=Path, default=OUTPUTS_DIR / "dataset_report.md")
    validate.set_defaults(func=validate_records)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
