# Before vs After: Base Model vs LoRA Adapter

Base model: `Qwen/Qwen2.5-1.5B-Instruct`  

Adapter: `/Users/jeffery/Desktop/research_ass/outputs/lora_adapter`

---

## test_01

**Input:**
> The softmax function converts a vector of real-valued scores into a probability distribution by exponentiating each element and normalizing by the sum of all exponentiated values. It is used as the final layer in multi-class classifiers and inside attention mechanisms.

### Base Model Output
```
Summary:
The softmax function maps real-valued scores to probabilities.

Key Points:
- Converts real-valued scores into probabilities.
- Used in multi-class classification.
- Inside attention mechanisms.

Limitation:
None identified.

Follow-up Question:
How does the softmax function handle ties in score comparison?
```

### LoRA Adapter Output
```
Summary:
The softmax function maps logits to probabilities that sum to one.

Key Points:
- The output of the softmax function lies between zero and one.
- Softmax preserves relative order among inputs with equal values.
- The derivative of softmax is easy to compute analytically.

Limitation:
Softmax can only handle vectors of positive numbers; negative numbers must be shifted before applying softmax.

Follow-up Question:
How does the choice of temperature affect the smoothness of the output?
```

---

## test_02

**Input:**
> Tokenization-free models like ByT5 operate directly on raw bytes rather than subword tokens. This eliminates tokenizer-induced biases and handles arbitrary multilingual text and code uniformly, at the cost of longer sequence lengths.

### Base Model Output
```
### Summary:
ByT5 operates on raw bytes without using subword tokenization, which helps eliminate bias from tokenizers and allows for handling multilingual and coded text uniformly.

### Key Points:
- **Operates Directly on Raw Bytes**: ByT5 processes input data as raw byte sequences.
- **Eliminates Tokenizer Biases**: The absence of subword tokenization reduces the impact of biased tokenization methods.
- **Handles Multilingual Text Uniformly**: Supports various languages and scripts through direct byte processing.
- **Supports Code Processing**: Can handle code snippets and other binary data types.
- **Longer Sequence Lengths**: May result in longer output sequences compared to models that use subword tokenization.

### Limitation:
The primary limitation is the increased length of the processed sequences due to the lack of subword tokenization, potentially leading to longer model training times and inference latency.

### Follow-Up Question:
How might researchers address the issue of longer sequence lengths when working with ByT5, especially considering potential trade-offs between accuracy and efficiency?
```

### LoRA Adapter Output
```
Summary:
ByT5 operates directly on byte-level input without tokenization.

Key Points:
- Byte-level processing allows handling arbitrary text/code sequences.
- Longer inputs result from padding to fixed length (e.g., 512 bytes).

Limitation:
Byte-level model outputs cannot be decoded back into human-readable text.

Follow-up Question:
How does byte-level output affect downstream tasks requiring sequential decoding?
```

---

## test_03

**Input:**
> The empirical risk minimization principle says learning algorithms should choose hypotheses that minimize the average loss on the training set. Structural risk minimization extends this by adding a complexity penalty to balance fit against model complexity.

### Base Model Output
```
### Summary:
Empirical Risk Minimization (ERM) is a fundamental approach in machine learning where models are selected based on minimizing the expected error over the entire dataset. Structural Risk Minimization (SRM), an extension of ERM, incorporates a penalty term for model complexity to prevent overfitting.

### Key Points:
1. **Empirical Risk Minimization**: Models chosen to minimize the average loss across the training data.
2. **Structural Risk Minimization**: Adds a penalty for model complexity to SRM, aiming to strike a balance between fitting the training data well and avoiding overly complex models prone to overfitting.

### Limitation:
While SRM addresses overfitting through complexity penalties, it may still struggle with high-dimensional spaces where the curse of dimensionality can lead to poor generalization performance.

### Follow-Up Question:
How might incorporating regularization techniques beyond just complexity penalties improve the robustness of structural risk minimization in high-dimensional settings?
```

### LoRA Adapter Output
```
Summary:
Empirical risk minimization chooses hypotheses with low expected error; structural risk minimization adds a penalty for complexity.

Key Points:
- The bias-variance tradeoff is inherent in both.
- Regularization parameters control the tradeoff between bias and variance.
- Empirical risk can be decomposed into bias and variance components.

Limitation:
Training overfits if regularization too weakly penalizes complexity.

Follow-up Question:
How does increasing data size affect generalization?
```

---

## test_04

**Input:**
> Causal language model pre-training on code (CodeX, StarCoder) teaches a model to complete program prefixes. Evaluation on HumanEval measures the fraction of generated code samples that pass unit tests, providing a functional correctness metric.

### Base Model Output
```
### Summary:
A causal language model is trained on code to generate and evaluate programs based on their functionality.

### Key Points:
- **Training Data**: CodeX and StarCoder datasets.
- **Objective**: To teach models how to complete program prefixes.
- **Evaluation Metric**: Fraction of generated code passing unit tests as a measure of functional correctness.

### Limitation:
The evaluation metric may not fully capture all aspects of program quality beyond just functional correctness.

### Follow-Up Question:
How can we improve the evaluation metric to better assess other qualities such as readability, maintainability, and efficiency?
```

### LoRA Adapter Output
```
Summary:
A causal language model trained on code learns to generate correct programs by predicting the next token conditioned on previous tokens and context.

Key Points:
- The model is fine-tuned with human-written test cases.
- Code completion accuracy correlates strongly with passing unit tests.
- The task requires understanding function signatures and calling conventions.

Limitation:
The dataset is biased towards common patterns found in existing codebases.

Follow-up Question:
How does the model generalize to novel functions not seen during training?
```

---

## test_05

**Input:**
> Adapter modules are small feed-forward bottleneck layers inserted between transformer sub-layers. Only adapter weights are trained during fine-tuning, keeping the pretrained backbone frozen and enabling modular, lightweight multi-task adaptation.

### Base Model Output
```
Summary:
Adapter modules are used to enable modular, lightweight multi-task adaptation by inserting small feed-forward bottleneck layers between transformer sub-layers.

Key Points:
- Adapter modules insert small feed-forward bottleneck layers.
- They are placed between transformer sub-layers.
- Only adapter weights are trained during fine-tuning.
- Pretrained backbone is kept frozen.
- Enables modular and lightweight multi-task adaptation.

Limitation:
The training of only adapter weights can lead to limited flexibility if the model needs to adapt to multiple tasks simultaneously.

Follow-up Question:
How does the freeze of the pretrained backbone affect the performance when the model needs to adapt to multiple tasks?
```

### LoRA Adapter Output
```
Summary:
Adapter modules insert small feed-forward bottleneck layers between transformer sub-layers to enable modular, lightweight multi-task adaptation without freezing the pretrained backbone.

Key Points:
- Adapter weights are only trained during fine-tuning.
- The pretrained backbone is kept frozen for efficiency.
- This approach enables fast transfer learning with minimal computational cost.

Limitation:
The training of adapter weights requires careful initialization and optimization strategies.

Follow-up Question:
How can we further improve the stability and performance of adapter modules across different tasks?
```

---

## test_06

**Input:**
> Curriculum learning trains a model on examples ordered from easy to hard, mimicking the way humans are taught. The hypothesis is that early exposure to simple patterns provides useful inductive biases that improve convergence on difficult examples later.

### Base Model Output
```
### Summary:
Curriculum Learning trains models sequentially by presenting them with progressively more complex data, hypothesizing that simpler initial exposures provide beneficial inductive biases leading to better performance on harder tasks.

### Key Points:
- **Sequential Training**: Models learn one task after another.
- **Increasing Complexity**: Data complexity increases as training progresses.
- **Inductive Biases**: Early exposure helps build useful inductive biases.
- **Improved Convergence**: Better generalization on challenging tasks.

### Limitation:
The effectiveness of curriculum learning can vary depending on the specific problem and dataset characteristics.

### Follow-Up Question:
How might incorporating domain-specific knowledge into the curriculum design further enhance its efficacy across different domains?
```

### LoRA Adapter Output
```
Summary:
Curriculum learning orders training data by difficulty and uses progressively harder tasks to train models.

Key Points:
- Early examples provide strong initial inductive bias.
- Harder examples help generalize better than random initialization.
- Curriculum learning can be applied to supervised, reinforcement, and semi-supervised settings.

Limitation:
Curriculum learning may not always converge faster; it depends on task complexity and model architecture.

Follow-up Question:
How does curriculum learning interact with other optimization techniques like gradient descent?
```

---

## test_07

**Input:**
> T5 reframes all NLP tasks as text-to-text problems, using the same encoder-decoder architecture and training objective (span denoising) across translation, summarization, classification, and question answering. A prefix in the input specifies the task.

### Base Model Output
```
### Summary:
T5 reinterprets various natural language processing (NLP) tasks such as translation, summarization, classification, and question answering by framing them as text-to-text problems.

### Key Points:
- T5 uses an encoder-decoder architecture.
- It employs the span denoising loss function for all tasks.
- The model is trained on diverse datasets including Wikipedia articles, books, and web pages.
- Tasks like translation, summarization, classification, and QA are unified under the framework of text-to-text generation.

### Limitation:
The approach may not generalize well to new domains outside its initial dataset due to limited exposure to unseen data types.

### Follow-Up Question:
How can T5 be expanded to handle more complex and varied NLP tasks beyond those initially covered?
```

### LoRA Adapter Output
```
Summary:
T5 treats all NLP tasks as text-to-text generation with shared encoder-decoder architecture.

Key Points:
- T5 uses span denoising loss for all tasks.
- Prefixes specify task type; e.g., "translate" for translation.
- Encoder is shared across tasks but decoder architectures differ.

Limitation:
No explicit cross-task fine-tuning; requires manual transfer learning.

Follow-up Question:
How can we improve cross-task generalization?
```

---

## test_08

**Input:**
> Monte Carlo Tree Search (MCTS) combines tree search with random simulation to estimate the value of game states. Used in AlphaGo, MCTS selects moves by balancing exploitation of high-value nodes with exploration of less-visited branches using the UCB formula.

### Base Model Output
```
### Summary:
Monte Carlo Tree Search (MCTS) is an algorithm that combines tree search and random simulation to estimate the value of game states.

### Key Points:
- **Combination**: MCTS integrates tree search with random simulation.
- **Estimation**: It estimates the value of game states through simulations.
- **Selection**: Uses the Upper Confidence Bound (UCB) formula to balance exploitation and exploration.
- **Application**: Implemented in AlphaGo for playing Go.

### Limitation:
The main limitation of MCTS is its computational complexity, which can be significant for large games like chess or Go due to the need for extensive simulations.

### Follow-Up Question:
How does the efficiency of Monte Carlo Tree Search (MCTS) compare to other algorithms when applied to complex board games such as chess or Go?
```

### LoRA Adapter Output
```
Summary:
MCTS balances tree search and random sampling to select optimal moves.

Key Points:
- It uses an MDP model to predict future rewards.
- The root node is expanded until a leaf node is reached.
- Random simulations explore unexplored branches; best actions are selected based on UCB estimates.

Limitation:
Exploration vs. exploitation trade-off can lead to suboptimal play if not balanced carefully.

Follow-up Question:
How does MCTS handle uncertainty about the true policy distribution?
```

---

## test_09

**Input:**
> Activation functions introduce nonlinearity into neural networks, enabling them to approximate arbitrary functions. ReLU replaced sigmoid and tanh as the dominant choice due to its simplicity and resistance to vanishing gradients, while variants like GELU and SiLU are now common in transformers.

### Base Model Output
```
**Summary:**
ReLU (Rectified Linear Unit) is widely used as an activation function in neural networks due to its simplicity and effectiveness in preventing vanishing gradients.

**Key Points:**
- **ReLU's Dominance:** It has become the most commonly used activation function.
- **Vanishing Gradients Problem:** ReLU helps mitigate issues related to gradient disappearance during backpropagation.
- **Alternative Functions:** Variants like GELU and SiLU have gained popularity for their improved performance in certain contexts.

**Limitation:**
While ReLU is effective, it may not perform well with very deep networks due to the "dying ReLU" problem.

**Follow-Up Question:**
What are some potential strategies to address the dying ReLU issue?
```

### LoRA Adapter Output
```
Summary:
ReLU introduces linear regions between activations, allowing neurons to learn complex representations without exploding gradients.

Key Points:
- ReLU's linearity allows fast backpropagation through hidden layers.
- It avoids the vanishing gradient problem of earlier activation functions.
- Variants like Leaky ReLU mitigate dead neuron issues by providing small positive slopes for negative inputs.

Limitation:
Gradient explosion can occur if the input is too large, leading to unstable training dynamics.

Follow-up Question:
How does ReLU's linearity affect downstream models' ability to generalize?
```

---

## test_10

**Input:**
> Optimal transport theory measures the distance between two probability distributions by finding the minimum-cost plan for transforming one distribution into another. It has been applied to generative modeling, domain adaptation, and training signal alignment in language models.

### Base Model Output
```
Summary:
Optimal transport theory is used to measure the distance between two probability distributions.

Key Points:
- Measures the cost of transporting mass from one distribution to another.
- Applied in generative modeling, domain adaptation, and signal alignment in language models.

Limitation:
- Requires knowledge of both input distributions.

Follow-up Question:
How can optimal transport be adapted to improve the performance of language models?
```

### LoRA Adapter Output
```
Summary:
Optimal transport measures the cost of transporting mass from one distribution to another.

Key Points:
- The Kantorovich-Wasserstein metric is defined as the infimum of expected costs over all couplings.
- Wasserstein barycenters represent the optimal mixture of distributions with given marginals.
- Optimal transport can be used to train contrastive loss functions that align representations across domains.

Limitation:
The curse of dimensionality makes exact computation intractable; approximate methods like Sinkhorn-Knopp scaling are often used instead.

Follow-up Question:
Can optimal transport be generalized to handle non-Euclidean spaces?
```

---
