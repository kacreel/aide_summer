# Lab 11: Can a Machine Explain Itself? (LIME, SHAP, and Saliency Maps)

**AIDE Summer 2025 | Duration: 120 minutes**

A lab on explainable AI written for philosophy students. No coding is required. Students run pre-written cells, move sliders, try their own inputs, and answer thinking questions. The technical content is kept light on purpose so the philosophical questions can carry the weight.

---

## Table of Contents
1. [Pre-Lab Learning](#pre-lab-learning)
2. [Lab Schedule](#lab-schedule)
3. [Learning Objectives](#learning-objectives)
4. [Key Concepts](#key-concepts)
5. [The Through-Line](#the-through-line)
6. [Quick Links](#quick-links)
7. [Discussion Questions](#discussion-questions)
8. [Teaching Notes](#teaching-notes)

---

## Pre-Lab Learning (30 minutes)

1. **Read (~15 min):**
   - [ProPublica, "Machine Bias" (2016)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing). The COMPAS investigation that Part 1 builds on. Students met this in the fairness labs; a re-skim is enough.

2. **Watch (~10 min):**
   - A short overview of explainable AI. Any clear introduction to "why did the model decide that" is fine. The lab itself teaches the three methods from scratch.

3. **Think (~5 min):**
   - Come with one sentence on this: when you explain why you did something, are you reporting the real cause of your action or telling a story that fits? Hold that thought; the lab returns to it three times.

No accounts, tokens, or local files are needed. The notebook installs everything and pulls its data from public sources.

---

## Pre-Lab Email

Send the day before. Swap any bracketed links for your own first.

```
Subject: Tomorrow: can a machine explain itself?

Hi everyone,

Last lab of the run, and it's built for you specifically. No coding. You run
cells, drag sliders, and argue with what comes out.

The topic is explainable AI. When a system decides something about a person,
whether they'll reoffend, what's in a photo, we now have tools that claim to
tell us why. We'll use three of them. Then we get to the question that's really
yours to answer: when the machine hands you a reason, should you believe it?
It's the old reasons-versus-causes problem, except the thing offering reasons is
a model that might be making them up after the fact.

About 30 minutes before class.

1. Re-skim the ProPublica "Machine Bias" piece from the fairness labs:
   https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
   We build straight on COMPAS.
2. Watch any clear short intro to explainable AI: [intro video link]. The lab
   teaches the methods from scratch, so this is just for flavor.
3. Come with one sentence on this: when you explain why you did something, are
   you reporting the real cause, or telling a story that happens to fit?

Setup is nothing. A Google login for Colab and you're set. No code to write, I
promise.

This one rewards talking, so come ready to disagree with each other.

See you tomorrow,
[name]
```

---

## Lab Schedule (120 minutes)

| Time | Section | Activity | Format |
|------|---------|----------|--------|
| 0–10 | **Framing** | Reasons vs causes; the confabulation worry; what the lab will and will not settle | Discussion |
| 10–15 | **Part 0: Setup** | Run the install and import cells | Press play |
| 15–50 | **Part 1: SHAP on COMPAS** | Per-person verdict breakdowns; the threshold slider and its uneven effect across groups | Demo + tinker + discuss |
| 50–75 | **Part 2: LIME on text** | Which words drove an atheism-vs-Christianity call; the confabulation parallel | Demo + tinker + discuss |
| 75–100 | **Part 3: Saliency and Grad-CAM** | Where an image model "looked"; perception vs metaphor; the Clever Hans problem | Demo + tinker + discuss |
| 100–120 | **Part 4: Synthesis** | The faithfulness problem; the five reflection prompts | Guided writing + discussion |

The three middle parts are independent. If time runs short, Part 3 can be cut without breaking anything.

---

## Learning Objectives

By the end of this lab, students will be able to:
- Explain in plain language what **SHAP**, **LIME**, and **saliency maps** each try to do, and on what kind of data
- Read a per-person SHAP breakdown and a LIME word-attribution chart without being misled by them
- Use a **decision threshold** as a worked example of a value judgement that no model can make for you
- State the **faithfulness problem**: an explanation can be plausible to a human and still misdescribe what the model did
- Connect post-hoc explanation to the philosophical distinction between a **reason** and a **cause**, and to debates about confabulation, perception, and what makes an explanation good

---

## Key Concepts

| Concept | Plain-language version |
|---------|------------------------|
| **Post-hoc explanation** | A story about why a model decided something, produced after the decision rather than baked into the model |
| **SHAP** | Splits a prediction into a fair share for each input fact, using Shapley values from game theory |
| **Shapley value** | A fair way to divide credit among contributors by averaging over every order they could have joined |
| **LIME** | Hides parts of an input and watches the guess shift, then tells a small local story about which parts mattered |
| **Saliency map** | A heat map over an image marking the pixels the decision was most sensitive to |
| **Grad-CAM** | A coarser image heat map that highlights whole regions the deep layers responded to |
| **Decision threshold** | The cutoff that turns a continuous score into a yes/no label; choosing it is a value judgement |
| **Faithfulness** | Whether an explanation actually matches the model's computation, as opposed to merely sounding right |
| **Reasons vs causes** | The cause of a behaviour and the reason offered for it can come apart; explanation tools give the offered reason |
| **Clever Hans / shortcut learning** | A model getting the right answer for the wrong reason, by reading a background cue rather than the thing itself |

---

## The Through-Line

The lab is built around one philosophical claim that gets stronger with each part:

1. **Part 1 (SHAP):** an explanation can be perfectly faithful to the model and still leave the real ethical question untouched. SHAP tells you race influenced a score. It cannot tell you whether it should.
2. **Part 2 (LIME):** an explanation can be a plausible story rather than the true cause, exactly as in the split-brain confabulation studies. A model can sort text correctly and "explain" itself with a word that was never doing real work.
3. **Part 3 (Saliency):** an explanation can show you where without showing you what. "The model looked at the ears" borrows a vocabulary of perception the pixels have not earned.
4. **Part 4 (Synthesis):** put together, these tools are necessary but not sufficient. They give us something to interrogate. They do not hand us permission to trust. An explanation that cannot be checked is a claim about a justification, not a justification.

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [Lab11_Explaining_AI_Decisions.ipynb](Scripts/Lab11_Explaining_AI_Decisions.ipynb) | The lab notebook. Runs end to end on free Colab; no GPU required. |
| [SHAP documentation](https://shap.readthedocs.io/) | The library used in Part 1 |
| [LIME repository](https://github.com/marcotcr/lime) | The library used in Part 2 |
| [Captum](https://captum.ai/) | The attribution library used in Part 3 |
| [ProPublica, "Machine Bias"](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) | The COMPAS investigation behind Part 1 |
| ["Why Should I Trust You?" (Ribeiro et al., 2016)](https://arxiv.org/abs/1602.04938) | The paper that introduced LIME |
| [A Unified Approach to Interpreting Model Predictions (Lundberg and Lee, 2017)](https://arxiv.org/abs/1705.07874) | The SHAP paper |

---

## Discussion Questions

These mirror the reflection prompts at the end of the notebook and work well out loud.

1. When SHAP, LIME, or Grad-CAM produces something that looks like a reason, is it a reason or a rationalisation? What evidence would settle it?
2. Who should choose the risk threshold in a real courtroom, and what would make their choice legitimate rather than arbitrary?
3. Does "the model looked at the ears" describe perception, or is it a metaphor we should drop?
4. Pick a philosophical standard for what makes an explanation good (Hempel, Salmon, or another) and judge the three tools against it.
5. Some laws grant a "right to an explanation" for automated decisions. After this lab, is that right worth much, and what would the explanation need to be for it to mean something?

---

## Teaching Notes

- **No coding background needed.** Reassure students up front. Their job is to run cells, move sliders, and think. The code is there to read if curious, not to write.
- **The threshold slider is the centerpiece of Part 1.** Give it real time. The moment students realise there is no neutral cutoff, and that the per-group flag rates move apart, is the moment the ethics lands. Let them sit in the discomfort rather than resolving it.
- **Encourage broken inputs in Part 2.** Ask students to write text they are sure is "obviously" about faith, then watch LIME point somewhere they did not expect. The gap is the lesson.
- **Expect the three image methods to disagree in Part 3.** That disagreement is a feature. It shows that "where the model looked" is a description we choose, not a fact we read off.
- **The data includes race as a feature on purpose.** Be ready to discuss why that choice was made for teaching and why a deployed tool's choice to include or drop it is itself an ethical position, not a technical default.
- **Runtime.** The notebook runs on a free Colab CPU runtime in a few minutes. A GPU is not required. The image model downloads a small set of pretrained weights the first time Part 3 runs.

---

## Navigation

**Previous Lab:** [← Lab 10 – Agent-Based Models of Scientific Communities](../Lab10-epistemic%20simulation/Lab10-lecture-guide.md)

---

**Author:** [Aniket Ghosh](https://www.linkedin.com/in/aniketghosh-/)
