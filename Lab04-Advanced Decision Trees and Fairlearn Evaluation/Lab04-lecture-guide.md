# Lab 4: Machine Learning Fairness with COMPAS - Student Guide

## 📋 What You'll Learn Today
- [Overview](#overview)
- [Before We Start](#before-we-start)
- [Session Schedule](#session-schedule)
- [What to Expect](#what-to-expect)
- [Resources](#resources)

## Overview
Welcome to the fairness in AI lab! Today you'll work with real criminal justice data to understand how machine learning can be both powerful and potentially harmful. You'll build models to predict recidivism (repeat offenses) and learn to evaluate whether they treat different groups fairly.

**Duration:** 2 hours  
**What you need:** Python fundamentals and machine learning basics from previous labs  
**Dataset:** COMPAS criminal risk assessment data

## Before We Start

### 📚 What You Should Know
- Python programming (variables, functions, data structures)
- Pandas for data manipulation
- Basic machine learning concepts (training, testing, accuracy)
- Decision trees from Lab 2.02

### 💻 Technical Setup
You'll need these libraries:
- pandas (data manipulation)
- scikit-learn (machine learning algorithms)  
- fairlearn (fairness metrics)
- matplotlib (visualization)

### 🎯 Learning Goals
By the end of today, you'll be able to:
- **Understand** the ethical implications of AI in criminal justice
- **Clean and prepare** real criminal justice data for analysis
- **Build** machine learning models for recidivism prediction
- **Evaluate** model fairness across racial groups using multiple metrics
- **Compare** different algorithms for accuracy vs. fairness trade-offs
- **Make informed decisions** about deploying AI systems responsibly


## Session Schedule

### 🚀 Part 1: Understanding the Problem (30 minutes)
**What we'll do:** Load and explore the COMPAS dataset
- Learn about COMPAS: what it is and why it's controversial
- Load and clean the criminal justice dataset
- Explore the target variable (recidivism) and demographic information
- Prepare features for machine learning

**You'll learn:** How to work with sensitive, real-world data and understand its context

### 🌳 Part 2: Building Your First Model (25 minutes)
**What we'll do:** Create a decision tree to predict recidivism
- Train a decision tree using criminal history features
- Split data into training and testing sets
- Make predictions and evaluate accuracy
- Compare your model with the original COMPAS algorithm

**You'll learn:** How to build and evaluate machine learning models on real data

### ⚖️ Part 3: Measuring Fairness (35 minutes)
**What we'll do:** Analyze whether your model treats different groups fairly
- Implement **Demographic Parity**: Do all racial groups get similar prediction rates?
- Calculate **False Positive Rates**: Are innocent people from different groups equally protected?
- Measure **Equalized Odds**: Does the model perform equally well across groups?
- Experiment with different model parameters to balance accuracy and fairness

**You'll learn:** How to quantify fairness and understand trade-offs with accuracy

### 🔄 Part 4: Comparing Different Approaches (25 minutes)
**What we'll do:** Try multiple machine learning algorithms
- Implement at least two different models (Random Forest, SVM, KNN, or Logistic Regression)
- Compare accuracy and fairness metrics across all models
- Create a summary table of model performance
- Decide which model you would recommend and why

**You'll learn:** How different algorithms perform on fairness vs. accuracy

### 🤔 Part 5: Reflection and Discussion (5 minutes)
**What we'll do:** Think critically about the broader implications
- Discuss accuracy vs. fairness trade-offs you observed
- Consider the ethical implications of your feature choices
- Reflect on real-world impact of false positives and negatives
- Suggest improvements for more fair and accurate models

**You'll learn:** How to think critically about deploying AI systems responsibly

### 🌟 Bonus Challenges (if you finish early)
Four optional extensions designed for students with a philosophy background:

1. **Feature Engineering** — create a "juvenile trajectory" feature; reflect on Aristotelian habituation and whether past behavior should determine future predictions
2. **Fairness Landscape Visualization** — use `MetricFrame` to plot FPR, FNR, TPR, and accuracy across all racial groups; engage with the Rawls vs. Nozick debate on distributive justice
3. **Intersectionality** — analyze fairness by age, sex, and Race×Sex simultaneously; apply Crenshaw's intersectionality framework to algorithmic bias
4. **Cross-Validation & Epistemic Humility** — use 10-fold CV to visualize accuracy variance; reflect on Humean induction, Kantian ends-vs-means, and practical wisdom (*phronesis*)

## What to Expect

### 📋 Exercise Structure

| Part | Focus | What You'll Code | Key Learning |
|------|-------|------------------|-------------|
| **Part 1** | **Data Exploration** | Load, clean, and explore COMPAS data | Understanding real-world dataset challenges |
| **Part 2** | **Model Building** | Train decision tree and evaluate accuracy | Basic ML workflow with sensitive data |
| **Part 3** | **Fairness Analysis** | Calculate fairness metrics across racial groups | Quantifying bias and fairness trade-offs |
| **Part 4** | **Model Comparison** | Implement and compare multiple algorithms | Finding the best approach for sensitive applications |
| **Bonus** | **Philosophy & Fairness** | Feature engineering, intersectionality, cross-validation | Connecting algorithmic fairness to ethical theory |

## Resources

### 📓 Today's Materials
| Resource | Description |
|----------|-------------|
| [COMPAS Exercises Notebook](Scripts/Lab04.COMPAS_exercises.ipynb) | **Your main workspace — TODOs with 💡 hints** |
| [COMPAS Completed Notebook](Scripts/Lab04.COMPAS_exercises_completed.ipynb) | **Reference solution — check your work here** |
| [COMPAS Dataset](Data/compas-scores-two-years.csv) | **Real criminal justice data for analysis** |

### 📚 Additional Context
- [Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) - Review decision trees concepts
- [FairLearn Documentation](https://fairlearn.org/v0.10/user_guide/) - Understanding fairness metrics
- [ProPublica COMPAS Investigation](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) - The journalism that exposed COMPAS bias

### 📖 Philosophical References (for Bonus Challenges)
- Aristotle, *Nicomachean Ethics* — habituation, character, and moral responsibility
- Rawls, *A Theory of Justice* (1971) — difference principle and the worst-off group
- Nozick, *Anarchy, State, and Utopia* (1974) — entitlement theory and historical justice
- Crenshaw, "Demarginalizing the Intersection of Race and Sex" (1989) — intersectionality
- Kant, *Groundwork for the Metaphysics of Morals* — treating persons as ends, not means
- Angwin et al., "Machine Bias" *ProPublica* (2016) — the original COMPAS investigation
- Chouldechova, "Fair Prediction with Disparate Impact" (2017) — impossibility theorem for fairness metrics


## 🗂️ Navigation
- [← Previous: Lab 3 - Fairness with COMPAS](../Lab03-Fairness%20with%20COMPAS/Lab03-lecture-guide.md)
- [← Back to Main Course](../README.md)
- [→ Next: Lab 5 - Differential Privacy](../Lab05-Differential%20Privacy/Lab05-lecture-guide.md)



