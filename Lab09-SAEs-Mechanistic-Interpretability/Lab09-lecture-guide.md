# Lab 9: Sparse Autoencoders & Mechanistic Interpretability

## 🚀 Open in Colab

| Version | Hardware | Model | Link |
|---------|----------|-------|------|
| **Normal** (default) | Free T4 | GPT-2 Small + Joseph Bloom's SAEs | [Open in Colab](https://drive.google.com/file/d/1pFY14lDww6qh9VrIMl-suFWsa3YOWASy/view?usp=sharing) |
| **A100** | Colab Pro A100 | Gemma 2 9B + DeepMind's Gemma Scope SAEs | [Open in Colab](https://drive.google.com/file/d/1jbiCM8RnKpNl6Hols6bTd4fnSbKZJx4L/view?usp=sharing) |

---

## Table of Contents
1. [Pre-Lab Learning](#pre-lab-learning)
2. [Lab Schedule](#lab-schedule)
3. [Learning Objectives](#learning-objectives)
4. [Key Concepts](#key-concepts)
5. [Quick Links](#quick-links)

---

## Pre-Lab Learning (30 minutes)

### Required Materials

1. **Read — intro + figures only (~15 min):**
   - [Towards Monosemanticity — Cunningham et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)  
     *Focus on*: the first two sections and all figures. Skip the math. Key claim: Sparse Autoencoders decompose polysemantic neurons into interpretable features.

2. **Watch (~15 min):**
   - [3Blue1Brown — What is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)  
     *Why:* Builds the vocabulary (neurons, activations, layers) needed before the lab.

3. **Pick your notebook (~1 min):**
   - **[Lab09_SAE.ipynb](Scripts/Lab09_SAE.ipynb)** — GPT-2 Small + Joseph Bloom's SAEs. Runs on **free Colab T4**. No HuggingFace account required. Features are noisier and steering effects subtler, but the workflow is identical. **Default choice for class.**
   - **[Lab09_SAE_A100.ipynb](Scripts/Lab09_SAE_A100.ipynb)** — Gemma 2 9B + DeepMind's Gemma Scope SAEs. Requires **Colab Pro A100** (paid) and HuggingFace access to the gated Gemma 2 9B model (see step 4). Substantially cleaner features and visibly stronger steering.

4. **Preview the notebook (~5 min):**
   - Skim Part 1 and the Neuronpedia instructions in Part 3 of whichever version you'll run.

5. **(A100 version only) Get Hugging Face access for Gemma (~5 min, do this before lab):**
   - Sign in to [huggingface.co](https://huggingface.co) (create account if needed)
   - Accept the license at [huggingface.co/google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) — instant approval
   - Create a read-only token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Save the token somewhere safe — the notebook will prompt for it

---

## Lab Schedule (120 minutes)

| Time | Section | Activity | Format |
|------|---------|----------|--------|
| 0–15 | **Part 1: Polysemanticity** | Toy superposition demo; reading the diagram | Demo + Guided reading |
| 15–25 | **Part 2: SAE Architecture** | Conceptual tour — how sparsity creates monosemanticity | Guided reading |
| 25–45 | **Part 3 & 4: Finding Features** | Load model + SAE (GPT-2 Small on T4, or Gemma 2 9B on A100); extract features on philosophical text | Demo + Exercise |
| 45–70 | **Part 5: Artifact Testing** | Cross-register test; PhilPapers; interpret results | Demo + Exercise |
| 70–100 | **Part 6: Causal Proof** | Steering (activation) + suppression (ablation); three-part causal test | Demo + Exercise |
| 100–120 | **Part 7: Synthesis** | What was proved, what wasn't; auditing implications | Guided reading |

---

## Learning Objectives

By the end of this lab, students will be able to:
- Explain why neural networks exhibit **polysemanticity** and what superposition predicts
- Describe the architecture and training objective of a **Sparse Autoencoder** without implementing one
- Use **SAELens** to extract feature activations from a pretrained model (GPT-2 Small + Joseph Bloom's SAEs, or Gemma 2 9B + DeepMind's Gemma Scope SAEs) on philosophical text
- Test whether features are genuine concept encodings or **dataset artifacts** using a cross-register analysis
- Prove **causal relevance** — not just correlation — using both **activation steering** and **suppression ablation**
- Articulate what the three-part causal test establishes and what it does not

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Polysemanticity** | A single neuron activates for multiple unrelated concepts — the problem SAEs solve |
| **Superposition** | Networks store more features than neurons via overlapping directions in activation space |
| **Sparse Autoencoder (SAE)** | Encoder–decoder trained with an L1 sparsity penalty; decomposes activations into sparse, interpretable features |
| **Monosemanticity** | A feature that corresponds to exactly one concept — verified by multi-corpus testing, not just Neuronpedia |
| **Dataset artifact** | A feature that *appears* monosemantic only because the test sentences share a common rhetorical register, not a common concept |
| **Cross-register test** | Measuring feature activation across informal, academic, legal, and literary versions of the same concept to rule out style artifacts |
| **Activation steering** | Adding a feature's decoder direction to the residual stream — tests whether the feature causally *promotes* the concept |
| **Suppression ablation** | Projecting out a feature's decoder direction from the residual stream — tests whether the feature causally *contributes* to the concept |
| **Convergent causal evidence** | When correlation + steering + suppression all point the same way; equivalent to the gain-of-function / loss-of-function standard in neuroscience |
| **SAELens** | Python library: `SAE.from_pretrained(...)` |
| **Neuronpedia** | Web browser for SAE features; useful for forming hypotheses, but labels are corpus-relative (Gemma Scope's training mix) |
| **Gemma Scope** | DeepMind's open SAE suite over every layer of Gemma 2 (2B/9B/27B); current gold standard for public interpretability research |

---

## The Causal Argument in Full

Finding a feature is not enough. The lab walks students through a three-step argument:

1. **Correlation** (Parts 3–5): Feature X activates on concept Y text, across multiple text registers and the PhilPapers corpus. This rules out a simple style artifact but does not establish causation.

2. **Activation test** (Part 6): Artificially *adding* Feature X's direction to the residual stream causes the model to generate more concept-Y-related output. The feature is causally sufficient to promote the concept.

3. **Suppression test** (Part 6): *Removing* Feature X's direction from the residual stream causes the model to generate less concept-Y-related output. The feature is a significant causal contributor.

When all three hold, this meets the same evidential standard neuroscience uses to attribute function to a brain region: correlation evidence, gain-of-function, and loss-of-function.

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [Lab09_SAE.ipynb](Scripts/Lab09_SAE.ipynb) | **Free-T4 notebook** — GPT-2 Small + Joseph Bloom's SAEs. Default for class. |
| [Lab09_SAE_A100.ipynb](Scripts/Lab09_SAE_A100.ipynb) | **A100 / Colab Pro notebook** — Gemma 2 9B + DeepMind's Gemma Scope SAEs. Cleaner features, visible steering. |
| [SAELens Docs](https://jbloomaus.github.io/SAELens/) | Python library for loading pretrained SAEs |
| [Gemma Scope Announcement](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) | DeepMind's open SAE suite on Gemma 2 — what powers the A100 version |
| [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | **Required reading** — skim intro + figures |
| [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | SAEs at scale on Claude 3 — optional deeper reading |
| [PhilPapers](https://philpapers.org) | Academic philosophy database — source of the Lab 02.01 dataset |

---

## Navigation

**Previous Lab:** [← Lab 8 – Gentle Hugging Face – Capabilities of LLMs](../Lab08-Gentle%20Hugging%20Face-Capabilities%20of%20LLMs/Lab08-lecture-guide.md)  
**Next Lab:** [Lab 10 – Context Engineering – (Graph)RAGged LLMs →](../Lab10-Context%20Engineering-(Graph)RAGged-LLMs/Lab10-lecture-guide.md)

---

**Author:** [Aniket Ghosh](https://www.linkedin.com/in/aniketghosh-/)
