# Lab 07.01: Auditing LLMs with the Logit Lens

## Overview

**Duration:** 60 minutes  
**Runtime:** Google Colab (free T4 GPU)  
**Prerequisites:** Lab 07 (Transformers and Embeddings Fundamentals)

This mini-lab introduces **mechanistic interpretability** — the practice of reverse-engineering
how language models process information internally. Students implement the **logit lens**
technique from scratch and use it to observe how GPT-2 builds up predictions layer-by-layer.

---

## Learning Objectives

By the end of this lab, students will be able to:

- Explain what the **residual stream** is and why it makes the logit lens possible
- **Implement the logit lens** using TransformerLens: `residual → ln_final → unembed → softmax`
- Interpret logit lens heatmaps (probability, rank, and KL divergence modes)
- Use **KL divergence** to identify the layer at which a model "commits" to its answer
- Articulate why interpretability tools matter for AI safety and ethics

---

## Session Schedule

### Intro (5 minutes)
- Recap: the residual stream from Lab 07
- Motivation: why do we care what happens *inside* a model?
- Preview the logit lens visualization

### Setup and Load (5 minutes)
- Install `transformer_lens`, load GPT-2 Small
- Run model with activation cache, inspect cache keys

### Exercise 1 — Single Layer Prediction (15 minutes)
Students implement `get_logit_lens_predictions()`:
1. Grab `cache["resid_post", layer_idx]`
2. Apply `model.ln_final()`
3. Apply `model.unembed()`

Then test it: observe how the top predicted token changes across layers 0 → 11.

### Visualization and Discussion (10 minutes)
- Run the provided `plot_logit_lens()` in `probs` and `ranks` modes
- Class discussion: which positions converge earliest? which are hardest?

### Exercise 2 — Correct-Token Probability (10 minutes)
Students compute P(correct next token) at each layer, plot the "confidence ramp".

### Exercise 3 — KL Divergence (10 minutes)
Students compute KL(P_final ‖ P_layer) to measure how quickly each position converges.

### Exploration and Reflection (5 minutes)
Students try one alternate text (plasma, philosophy, or repetitive) and answer
reflection questions 1–5.

---

## Materials

| File | Description |
|------|-------------|
| [Scripts/Lab07.01_LogitLens.ipynb](./Scripts/Lab07.01_LogitLens.ipynb) | Student notebook — 3 exercises with `# YOUR CODE HERE` |
| [Scripts/Lab07.01_LogitLens_COMPLETED.ipynb](./Scripts/Lab07.01_LogitLens_COMPLETED.ipynb) | Instructor reference — all exercises solved |
| [Tutorials/interpreting GPT the logit lens — LessWrong.md](./Tutorials/interpreting%20GPT%20the%20logit%20lens%20—%20LessWrong.md) | Background reading: the original logit lens blog post |
| [Tutorials/What Is ChatGPT Doing.md](./Tutorials/What%20Is%20ChatGPT%20Doing.md) | Supplementary reading on transformer internals |

---

## Colab Setup

The notebook installs one dependency:

```
!pip install transformer_lens -q
```

All models load from Hugging Face (no authentication required).
GPT-2 Small (124M parameters) fits comfortably on a free T4 GPU.
No local files are read — the notebook is fully self-contained.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Residual stream** | The vector every layer reads from and writes to; accumulates information top-to-bottom |
| **Logit lens** | Decoding the residual stream at each intermediate layer: `ln_final → unembed → softmax` |
| **KL divergence** | Measure of how different two distributions are; used to find when a layer "converges" |
| **Mechanistic interpretability** | Field of reverse-engineering what computations a model performs |

---

## Discussion Questions

1. GPT-2 has 12 layers. At which layer does the model typically commit to its final answer for factual completions vs. ambiguous completions? What does this tell us about how "difficulty" is distributed across the network?

2. The logit lens reveals that some token positions are "resolved" very early (low KL by layer 3–4) while others stay uncertain until the final layer. What kinds of tokens tend to be resolved early?

3. If a future, more capable AI system made a harmful prediction, would the logit lens help us understand *why* it made that prediction? What would we learn? What would we miss?

4. Mechanistic interpretability researchers like to say they want to "read the model's thoughts." Is that a reasonable description of what the logit lens shows us? What are the limits of this metaphor?

---

## Navigation

**Previous:** [← Lab 07 — Transformers and Embeddings Fundamentals](../Lab07-Transformers%20and%20Embeddings%20Fundementals/Lab07-lecture-guide.md)  
**Next:** [Lab 08 — Hugging Face LLM Capabilities →](../Lab08-Gentle%20Hugging%20Face-Capabilities%20of%20LLMs/Lab08-lecture-guide.md)
