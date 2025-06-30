# Class 6: Regression Trees and LLM Concepts - Lecture Guide

## Table of Contents
1. [Pre-Class Learning](#pre-class-learning)
2. [Class Schedule](#class-schedule)
3. [Learning Objectives](#learning-objectives)
4. [Key Concepts](#key-concepts)
5. [Quick Links](#quick-links)

## Pre-Class Learning (2 hr 45 minutes)
### Required Materials:
1. **[Coursera Machine Learning with Python](https://www.coursera.org/learn/machine-learning-with-python?specialization=ibm-data-science)** (45 minutes)
   - Complete Module 3 including hands-on exercises
   - Optional: Regression Trees section

2. **[3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** (1 hr 10 minutes)
   - Visual introduction to neural network concepts
   - Mathematical intuition behind neural networks

3. **Blog Post** (40 minutes)
   - "[What is ChatGPT Doing…And Why Does it Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)" | [📁 Local File](./Tutorials/What%20Is%20ChatGPT%20Doing%20…%20and%20Why%20Does%20It%20Work?.md)
   - Understanding transformer architecture and LLM behavior

## Class Schedule

### Intro (5 minutes)
- Review pre-class materials
- Overview of neural networks and transformers

### Session 1: Perceptron by Hand (20 minutes)
**Foundational Neural Network Concepts**
- **Resource:** [W3Schools Perceptrons](https://www.w3schools.com/ai/ai_perceptrons.asp)
- **Exercise:** [📁 Perceptron Practice PDF](./Exercises/PERCEPTRON%20PRACTICE.pdf) | [☁️ Should we go to the beach?](https://docs.google.com/document/d/1DO568DRaTakcIdWXkmlHw_XWUQNPqQru_66E8YS5Yw0/edit?usp=sharing)
- Simple and straightforward introduction
- Manual calculation of perceptron outputs
- Transition to neural networks

### Session 2: Simple Neural Network Math by Hand (30 minutes)
**Manual Calculations**
- **Material:** [Introduction to Math Behind Neural Networks](https://towardsdatascience.com/introduction-to-math-behind-neural-networks-e8b60dbbdeba)
- Step-by-step forward propagation
- Weight updates and backpropagation basics
- Understanding activation functions

### Break (5 minutes)

### Session 3: Transformer by Hand - Core Concepts (35 minutes)
**Attention is All You Need**
- **Primary Tutorial:** [📁 Deep Dive into Transformers by Hand](./Tutorials/Deep%20Dive%20into%20Transformers%20by%20Hand%20✍︎.md) | [☁️ Google Doc Version](https://docs.google.com/document/d/12Y4gtQuzSpXj-pQLKJr6SrANs_oe9uVhBzjJ2mD0zjI/edit?usp=sharing)
- **Exercise:** [📁 Transformer Blank Exercise PDF](./Exercises/Transformer%20Blank%20Exercise.pdf)
- **Supporting Resource:** [Matrix Multiplication Guide](https://www.mathsisfun.com/algebra/matrix-multiplying.html)
- Manual calculation of attention mechanisms
- Understanding query, key, and value matrices

### Session 4: Advanced Attention (Bonus) (20 minutes)
**For Math-Savvy Students**
- **Advanced Tutorial:** [📁 Deep Dive into Self-Attention by Hand](./Tutorials/Deep%20Dive%20into%20Self-Attention%20by%20Hand✍︎.md) | [☁️ Google Doc Version](https://docs.google.com/document/d/1i1XEISzYFbydbixtxZfVpTn8Q0NPoP4lKlz6e8BWEhc/edit?usp=sharing)
- **Exercise:** [📁 Self-Attention Blank Exercise PDF](./Exercises/Self-Attention%20Blank%20Exercise.pdf)

## Learning Objectives
By the end of this class, students will be able to:
- **Neural Network Fundamentals:** Understand perceptrons and basic neural network architecture
- **Mathematical Foundation:** Perform forward propagation calculations by hand
- **Transformer Architecture:** Grasp attention mechanisms and transformer concepts
- **LLM Understanding:** Comprehend how large language models process information
- **Mathematical Intuition:** Develop deeper understanding through manual calculations

## Key Concepts

| Concept | Description | Application |
|---------|-------------|-------------|
| **Perceptron** | Simplest neural network unit | Binary classification, foundation for neural networks |
| **Forward Propagation** | Data flow through network layers | Prediction generation in neural networks |
| **Activation Functions** | Non-linear transformations | Sigmoid, ReLU, enabling complex patterns |
| **Attention Mechanism** | Focus on relevant input parts | Core of transformer architecture |
| **Query, Key, Value** | Attention computation matrices | Information retrieval and weighting |
| **Transformer Architecture** | Modern neural network design | Language models, BERT, GPT |
| **Self-Attention** | Relating positions within sequence | Understanding context and relationships |

## Quick Links

### 📚 Exercise Files
| Exercise | Local File | Online Version |
|----------|------------|----------------|
| Perceptron Practice | [📁 PDF](./Exercises/PERCEPTRON%20PRACTICE.pdf) | [☁️ Google Doc](https://docs.google.com/document/d/1DO568DRaTakcIdWXkmlHw_XWUQNPqQru_66E8YS5Yw0/edit?usp=sharing) |
| Transformer Exercise | [📁 PDF](./Exercises/Transformer%20Blank%20Exercise.pdf) | - |
| Self-Attention Exercise | [📁 PDF](./Exercises/Self-Attention%20Blank%20Exercise.pdf) | - |
| Vector Database Exercise | [📁 PDF](./Exercises/Vector%20Database%20Exercise%20Page.pdf) | - |

### 📖 Tutorial Materials
| Resource | Local File | Online Version | Format |
|----------|------------|----------------|---------|
| Transformers by Hand | [📁 Markdown](./Tutorials/Deep%20Dive%20into%20Transformers%20by%20Hand%20✍︎.md) | [☁️ Google Doc](https://docs.google.com/document/d/12Y4gtQuzSpXj-pQLKJr6SrANs_oe9uVhBzjJ2mD0zjI/edit?usp=sharing) | Tutorial |
| Self-Attention by Hand | [📁 Markdown](./Tutorials/Deep%20Dive%20into%20Self-Attention%20by%20Hand✍︎.md) | [☁️ Google Doc](https://docs.google.com/document/d/1i1XEISzYFbydbixtxZfVpTn8Q0NPoP4lKlz6e8BWEhc/edit?usp=sharing) | Advanced Tutorial |
| Vector Databases by Hand | [📁 Markdown](./Tutorials/Deep%20Dive%20into%20Vector%20Databases%20by%20Hand%20✍︎.md) | - | Bonus Material |
| Keys, Queries, Values Explained | [📁 Markdown](./Tutorials/What%20exactly%20are%20keys,%20queries,%20and%20values%20in%20attention%20mechanisms?.md) | [☁️ Stack Exchange](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms) | Reference |
| ChatGPT Explanation | [📁 Markdown](./Tutorials/What%20Is%20ChatGPT%20Doing%20…%20and%20Why%20Does%20It%20Work?.md) | [☁️ Stephen Wolfram](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) | Blog Post |

### 🔗 External Resources
| Resource | Description | Format |
|----------|-------------|---------|
| [W3Schools Perceptrons](https://www.w3schools.com/ai/ai_perceptrons.asp) | Basic perceptron introduction | Online Tutorial |
| [Neural Network Math](https://towardsdatascience.com/introduction-to-math-behind-neural-networks-e8b60dbbdeba) | Mathematical foundations | Blog Post |
| [Matrix Multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html) | Mathematical prerequisite | Reference |

### Optional Supplementary Resources:
- [📁 CNN Tutorial](./Tutorials/What%20is%20a%20Convolutional%20Neural%20Network%20(CNN)%20%20Definition%20from%20TechTarget.md) - For computer vision context
- [📁 GPT Logit Lens](./Tutorials/interpreting%20GPT%20the%20logit%20lens%20—%20LessWrong.md) - Advanced interpretability
- [Northeastern SEED Grant Projects](https://idi.provost.northeastern.edu/seed-grant-projects/) - Research context
- [Byron Wallace Research](https://www.byronwallace.com/) - Additional ML resources

## Navigation
**Previous Class:** [← Class 5 - Differential Privacy](../class5-differential-privacy/class5-lecture-guide.md)  
**Next Class:** [→ Class 7 - Capstone Project](../class7-capstone/class7-lecture-guide.md)