# Compute Where It Counts: High Quality Sparsely Activated LLMs


**CWIC** (Compute Where It Counts), is a new method for creating efficient transformers that automatically decide when to use more or less compute. CWIC makes models **faster, more cost-efficient, and more interpretable**.

**Summary:**

1. CWIC yields a 3x increase in CPU throughput with only a 10% reduction in benchmark performance.

2. CWIC uses a different amount of compute for each token, making task difficulty interpretable.

3. CWIC directly optimizes compute as a loss function, and learns to budget compute without labelled data or hand-crafted heuristics.

4. The CWIC architecture uses learned activation thresholds and expressive sparsity patterns to enable adaptive computation.

Read more [on our blog](https://crystalai.org/blog/2025-08-18-compute-where-it-counts)!

## Installation
```sh
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc

pixi shell
```

## Inference
```sh
python cwic_scripts/chat_with_it.py
```
This will let you chat with one of our models and get output highlighted based off active parameters used!

darker blue is more compute, light highlight is less
![chat_with_it.png](assets/chat_with_it.png)
## Training
```sh
wandb login
# gcloud auth application-default login --no-launch-browser # for saving checkpoints to google cloud

python cwic_scripts/train_cwic.py
```


## Background


Large language models have become ubiquitous tools for natural language tasks. However, LLM inference requirements have grown beyond consumer devices and drive massive industry hardware expenses. For many applications, especially agentic ones, inference speed and cost are critical bottlenecks for real world deployment.

Therefore, many methods have been proposed to improve LLM inference efficiency. These include [quantization](https://arxiv.org/abs/2402.16775), [pruning](https://arxiv.org/abs/2305.11627), and [sparse Mixture of Experts (MoE)](https://arxiv.org/abs/1701.06538). Activation sparsity, the category in which CWIC falls, is another such approach. It focuses on removing small and inconsequential activations from the inputs of matrix multiplications, allowing some computations to be skipped without affecting the model's output.

One of the earliest activation sparsity methods for LLMs was [Relufication](https://arxiv.org/abs/2310.04564), which inserted ReLU activation functions into LLMs to induce sparsity. [ProSparse](https://arxiv.org/abs/2402.13516v4) further increased sparsity by adding an L1 penalty to the ReLU activations. [Deja Vu](https://arxiv.org/abs/2310.17157) and [ShadowLLM](https://arxiv.org/abs/2406.16635) predicted sparsity on the fly by training small auxiliary MLPs. [Q-Sparse](https://arxiv.org/abs/2407.10969) discarded all but the top-K largest activations, and demonstrated a *sparse scaling law* where larger models are more robust to sparsity.

Most similar to our work are [CATS](https://arxiv.org/abs/2404.08763), [TEAL](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models), and [R-SPARSE](https://arxiv.org/abs/2504.19449). These methods all remove activations with smaller magnitude than a *threshold*. However, none of these methods directly learn activation thresholds. Furthermore, these methods suffer from performance collapse at high sparsity levels. CWIC addresses both limitations.


## Motivating Insights

1. Learned parameters perform better than heuristically chosen ones. The often-quoted ["bitter lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) states that general learning methods have historically outperformed hand-crafted approaches. We noticed that previous activation sparsity methods like [TEAL](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models) (block-wise greedy optimization) and [R-Sparse](https://arxiv.org/abs/2504.19449) (search algorithm) used heuristics to determine activation thresholds. We hypothesized that learning thresholds directly through backpropagation would lead to better results.

2. Adaptive computation methods with higher combinatorial expressiveness perform better. This was theorized and demonstrated by [DeepSeekMoE](https://arxiv.org/abs/2401.06066), which improved over previous MoE methods by increasing the number of experts to choose from. We posited that the same principle would apply to activation sparsity: sparsity patterns with higher flexibility than the standard column pattern would yield better performance.

3. Different parameters should have different sparsity levels. This insight was drawn from our own preliminary experiments. We found that, among other patterns, the Q attention matrix was more robust to sparsity than the K and V matrices. This shows a limitation in methods like [CATS](https://arxiv.org/abs/2404.08763) and [Q-Sparse](https://arxiv.org/abs/2407.10969) that use the same sparsity level for every parameter. Furthermore, while the sparsity level of each parameter could be manually tuned, we wanted to automate this by making sparsity thresholds learnable.

4. Easier problems should require less compute. As discussed in [Dynamic Routing in MoE Models](https://arxiv.org/abs/2403.07652), it is intuitively obvious that some outputs should have simpler derivations, and therefore should need less compute. This is exemplified by [GPT-5](https://openai.com/index/introducing-gpt-5/), which routes some inputs to a less costly model. We wanted to see if a sparsely activated model could learn this behavior on its own.

