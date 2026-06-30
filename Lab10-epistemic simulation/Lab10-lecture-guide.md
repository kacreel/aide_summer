# Lab 10: Agent-Based Models of Scientific Communities

**AIDE Summer 2025 | Duration: 120 minutes**

---

## Table of Contents
1. [Pre-Lab Learning](#pre-lab-learning)
2. [Lab Schedule](#lab-schedule)
3. [Learning Objectives](#learning-objectives)
4. [Key Concepts](#key-concepts)
5. [The Two-Part Argument](#the-two-part-argument)
6. [Quick Links](#quick-links)
7. [Common Student Questions](#common-student-questions)
8. [Assessment Ideas](#assessment-ideas)
9. [Instructor Notes](#instructor-notes)
10. [References](#references)
11. [Navigation](#navigation)

---

## Pre-Lab Learning (20 minutes)

### Required Materials

1. **Read — intro + figures only (~10 min):**
   - [The Communication Structure of Epistemic Communities — Zollman, 2007](https://doi.org/10.1086/524787)  
     *Focus on*: the abstract, the network diagrams, and the main result (Figures 1–2). Skip the formal proofs. Key claim: sparse communication networks outperform dense ones when evidence is ambiguous.

2. **Read — abstract + figures (~10 min):**
   - [Better than Best — Wu, 2019](https://doi.org/10.1086/703717)  
     *Focus on*: the NK landscape setup (Section 2), the "better vs. best" distinction, and Figure 3. Skip the simulation details — you'll implement them. Key claim: mixed communities outperform homogeneous ones on rugged fitness landscapes.

3. **Pick your starting notebook (~1 min):**
   - **[Lab11A_MultiArmBandit.ipynb](Scripts/Lab11A_MultiArmBandit.ipynb)** — Multi-arm bandits; the Zollman Effect. Run first.
   - **[Lab11B_NKLandscape.ipynb](Scripts/Lab11B_NKLandscape.ipynb)** — NK landscapes; epistemic diversity. Run second.
   - Both notebooks run on free Colab T4 (no GPU required).

4. **Preview the notebook (~5 min):**
   - Skim Part 1 of whichever notebook you start with. The opening markdown explains the model before any code runs.

---

## Pre-Lab Email

Send the day before. Swap any bracketed links for your own first.

```
Subject: Tomorrow: simulating how science actually makes progress

Hi all,

Lab 10 tomorrow. We step back from single models and ask a bigger question: how
does a community of scientists, each one stubborn, partial, and working with
patchy information, manage to get anything right? We'll build small simulations
of scientific communities and watch what happens when you change how they talk
to each other.

One result caught me off guard the first time I saw it. Sometimes a community
that shares information less freely reaches the truth more often. We'll work out
why that happens.

About 30 minutes to prepare.

1. Read the short framing on the Zollman effect in the lecture guide. That's the
   surprising result I just mentioned.
2. Want the original argument? Skim a Zollman paper (2007 or 2010, both linked in
   the guide). The introduction alone is plenty.
3. Open the notebook and run the first couple of cells so your Colab is awake:
   [notebook Colab link]

No accounts, no installs, no heavy math. Bring your gut sense of how a group
should be organized to find the truth. The lab is going to lean on it.

See you there,
[name]
```

---

## Lab Schedule (120 minutes)

| Time | Section | Activity | Format |
|------|---------|----------|--------|
| 0–15 | **Opening** | SSRI/ketamine case study; explore vs. exploit at the community level | Discussion |
| 15–50 | **Notebook 10A** | Bayesian agents + multi-arm bandit simulation; the Zollman Effect | Demo + Exercise |
| 50–60 | **Break + Discussion** | Transient diversity; real institutions that slow communication | Group discussion |
| 60–105 | **Notebook 10B** | NK landscapes; "Better than Best"; mixed-community advantage | Demo + Exercise |
| 105–120 | **Closing** | AI bridge; does AI summarization make the Zollman problem better or worse? | Discussion |

---

## Learning Objectives

By the end of this lab, students will be able to:
- Explain the **Zollman Effect** and why dense communication networks can hinder collective discovery when evidence is ambiguous
- Describe the **NK landscape** model and how the parameter K controls landscape ruggedness
- Implement **Bayesian belief updating** for agents sharing experimental results across a network
- Implement the **"Better" update strategy** (copy a random better peer, not just the best) and explain why it differs from "Best"
- Run **Monte Carlo simulations** and interpret convergence outcomes statistically
- Connect simulation results to real debates in science policy (preregistration, blind review, funding diversity)

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Explore-exploit tradeoff** | At the individual level, exploiting the best-known option is rational. At the community level, premature exploitation can lock everyone out of the true optimum |
| **Zollman Effect** | On a complete (dense) network, early chance variation propagates too fast — the community can lock in on the *wrong* research method even when all agents are perfectly rational Bayesians |
| **Epistemic diversity** | Maintaining a variety of research approaches as collective insurance against premature convergence |
| **Multi-arm bandit** | A model where agents repeatedly choose between options (research methods) with unknown payoffs, updating beliefs from noisy outcomes |
| **Bayesian updating** | Revising a Beta(α, β) belief about a method's success rate after observing new trials: `belief = (1 + hits) / (2 + hits + misses)` |
| **Network topology** | How scientists are connected: *Complete* (all-to-all), *Cycle* (ring of neighbors), *Wheel* (hub + ring) |
| **NK landscape** | A fitness function over binary strings of length N where each bit's contribution depends on K other bits. K=0: smooth, one peak. K≈N: maximally rugged, many local peaks |
| **Ruggedness** | Number of local optima on a fitness landscape. Increases with K. Hill-climbing gets stuck when ruggedness is high |
| **"Best" strategy** | Copy the single highest-fitness peer if they outperform you. Fast convergence; prone to locking in on local optima |
| **"Better" strategy** | Copy any better-performing peer, chosen at random. Slower convergence; maintains diversity across the landscape |
| **Mixed community** | A community where some agents use "Best" and some use "Better." Outperforms both homogeneous types on rugged landscapes (K ≥ 4) |
| **Individual vs. collective rationality** | Individually rational behavior can produce collectively suboptimal outcomes — a theme from Lab 3 (COMPAS) revisited here |

---

## The Two-Part Argument

These notebooks establish a two-part claim about science and rationality.

**Part 1 — Notebook 10A (Zollman)**: Even when every individual agent is a perfect Bayesian reasoner, the *structure of communication* determines whether the community finds the truth. A dense (complete) network propagates early evidence so fast that everyone updates together — amplifying early noise into permanent false consensus. A sparse (cycle) network insulates subgroups from each other long enough for the true signal to emerge. This is not a story about irrationality; it's about how rational local behavior can fail globally.

**Part 2 — Notebook 10B (Wu)**: On problems with *epistatic structure* — where changing one assumption changes the payoff of others (true of most real scientific problems) — mixed exploration strategies dominate homogeneous ones. "Best" agents climb fast but converge to local peaks. "Better" agents maintain spread but lose climbing force. A mixed community inherits both properties. The advantage is strongest when the landscape is rugged (K=4–7) and vanishes on smooth landscapes (K=0–2).

Together, these results suggest that epistemic diversity is not merely a nice-to-have but a structural necessity for collective truth-seeking on hard problems.

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [Lab11A_MultiArmBandit.ipynb](Scripts/Lab11A_MultiArmBandit.ipynb) | **Student notebook** — Multi-arm bandits (Zollman Effect). Start here. |
| [Lab11A_MultiArmBandit_COMPLETED.ipynb](Scripts/Lab11A_MultiArmBandit_COMPLETED.ipynb) | **Instructor answer key** — Full solution + extended analysis (network zoo, community size sweep, heatmaps, Mill's dissenter) |
| [Lab11B_NKLandscape.ipynb](Scripts/Lab11B_NKLandscape.ipynb) | **Student notebook** — NK landscapes (epistemic diversity). Run second. |
| [Lab11B_NKLandscape_COMPLETED.ipynb](Scripts/Lab11B_NKLandscape_COMPLETED.ipynb) | **Instructor answer key** — Full solution + K sweep |
| [Zollman 2007](https://doi.org/10.1086/524787) | **Required reading** — The Communication Structure of Epistemic Communities |
| [Wu 2019](https://doi.org/10.1086/703717) | **Required reading** — Better than Best |
| [dubova-et-al-2026.pdf](dubova-et-al-2026.pdf) | Optional deeper reading — theory-motivated vs. random experimentation |

---

## Common Student Questions

**"Is the Zollman effect just a simulation artifact?"**  
No. There is experimental evidence for communication-speed trade-offs in human lab settings (Derex & Boyd 2016). The intuition also matches the replication crisis literature — fields with fast, highly visible trial results show more replication failures than slower fields.

**"If the cycle network is better, why don't scientists actually communicate less?"**  
Individual incentives push toward maximum visibility — you want your work read, cited, and responded to. The social optimum and the individual optimum diverge. This is the same structure as the fairness-accuracy tradeoff in Lab 3.

**"Is there a real experiment where a mixed community outperformed a homogeneous one?"**  
The closest real case is drug discovery. Pharmaceutical companies that maintained parallel research programs across multiple mechanisms performed better long-term than those that converged on one mechanism early (Pammolli et al. 2011).

**"What's the connection to AI alignment?"**  
If we want AI systems that are genuinely robust across diverse contexts, we need the *community of AI researchers* to explore diverse paradigms — not converge on a single architectural approach. The same explore/exploit tension applies to research portfolios.

**"Does the Zollman Effect apply to AI trained on scientific literature?"**  
Yes, and it's an open question whether it makes things better or worse. LLMs trained on published science inherit whatever convergence biases existed in that literature. AI summarization tools could further accelerate premature consensus — or they could resurface buried minority approaches. The mechanism depends on which papers the model overweights.

---

## Assessment Ideas

- **Short response**: "A pharmaceutical company argues that all their labs should follow the most promising compound to maximize speed to market. Using the Zollman model, explain when this strategy fails and what alternative would be better."
- **Simulation extension**: Add a third research method and measure how network topology affects discovery of the globally best method.
- **Paper reading**: Assign either Wu (2019) or Zollman (2007) and ask students to identify one assumption in the model they would want to relax, and explain what effect relaxing it might have.

---

## Instructor Notes

- The Monte Carlo simulations take 2–3 minutes per cell on free Colab. Warn students to start runs before reading the next markdown section.
- For Exercise 1 in Notebook 10B, nudge students who write something like `update_best` — the key difference is `np.random.choice(better_indices)` rather than `np.argmax`. The randomness is the point.
- The `NKLandscape.global_max()` method uses exhaustive search, feasible only for N ≤ 18. For N=10 it runs in under 1 second.
- The `BtB_codes_sharable.py` file is the original simulation code from Wu's paper (Python 2.7 syntax — `print` without parentheses). It will not run directly; it's provided as reference only.
- The `science_cliques/` folder contains a third model (Zollman's testimonial norms / "Science Cliques") — a useful extension for advanced students.
- Both notebooks are independent. They can be taught sequentially in one 120-minute session or split across two shorter ones.

---

## References

1. Zollman, K. J. S. (2007). The communication structure of epistemic communities. *Philosophy of Science*, 74(5), 574–587.
2. Zollman, K. J. S. (2010). The epistemic benefit of transient diversity. *Erkenntnis*, 72(1), 17–35.
3. Wu, J. (2019). Better than best: Epistemic landscapes and diversity of practice in science. *Philosophy of Science*, 86(2), 333–358.
4. Weisberg, M., & Muldoon, R. (2009). Epistemic landscapes and the division of cognitive labor. *Philosophy of Science*, 76(2), 225–252.
5. Kauffman, S. A. (1993). *The Origins of Order*. Oxford University Press.
6. Dubova, M., et al. (2026). Against theory-motivated experimentation. *Collective Intelligence* (preprint in this directory).

---

## Navigation

**Previous Lab:** [← Lab 9 – Sparse Autoencoders & Mechanistic Interpretability](../Lab09-SAEs-Mechanistic-Interpretability/Lab09-lecture-guide.md)  
**Next Lab:** [Lab 11 – Can a Machine Explain Itself? (LIME, SHAP, Saliency Maps) →](../Lab11-Explainable%20AI-LIME%20SHAP%20Saliency/Lab11-lecture-guide.md)


--

**Author:** [Aniket Ghosh](https://www.linkedin.com/in/aniketghosh-/)
