# Lab 3: Fairness with COMPAS - Lecture Guide

## 📋 Table of Contents
- [Overview](#overview)
- [Pre-Lab Learning](#pre-lab-learning)
- [Lab Schedule](#lab-schedule)
- [Learning Objectives](#learning-objectives)
- [Quick Links](#quick-links)

## Overview
Exploration of fairness in machine learning through hands-on analysis using Google's What-If Tool and the controversial COMPAS criminal risk assessment dataset.

## Pre-Lab Learning (45 minutes)
### Required Materials:
1. **Background Reading** (25 minutes)
   - [Machine Bias (ProPublica)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) - The original investigation that exposed COMPAS bias
   
2. **Technical Preparation** (15 minutes)
   - [What-If Tool Overview](https://pair-code.github.io/what-if-tool/) - Skim the main page and features
   - [Fairness Metrics Explained](https://developers.google.com/machine-learning/glossary/fairness) - Basic definitions
   
3. **Video** (5 minutes)
   - [Algorithmic Bias Explained in 5 Minutes](https://www.youtube.com/watch?v=gV0_raKR2UQ) - Quick visual introduction

## Lab Schedule

### Session 1: What-If Tool Introduction (30 minutes)
**Understanding Algorithmic Fairness**
- **Tool:** [What-If Tool Tutorial](https://pair-code.github.io/what-if-tool/learn/tutorials/walkthrough/)
- **Demo:** [Census Dataset Walkthrough](https://pair-code.github.io/what-if-tool/demos/uci.html)
- Introduction to fairness concepts and bias detection
- Interactive model analysis techniques

### Session 2: COMPAS Dataset Analysis (40 minutes)
**Hands-on Bias Detection**
- **Tool:** [What-If Tool COMPAS Demo](https://pair-code.github.io/what-if-tool/demos/compas.html)
- **Worksheet:** [Analysis Worksheet](https://docs.google.com/document/d/1XRQTGwyYw6vZ4e2LGDVSM8BEfbxoY16BLYA5OnLF8O8/edit?usp=sharing)
- Analyze criminal risk assessment bias
- Explore fairness metrics and demographic disparities
- Document findings using structured analysis

### Session 3: Discussion & Implications (20 minutes)
**Critical Analysis**
- Share findings from COMPAS analysis
- Discuss real-world implications of biased algorithms
- Explore fairness-accuracy trade-offs

## Learning Objectives
Students will learn to:
- **Understand** algorithmic fairness concepts and bias sources
- **Use** Google's What-If Tool for interactive model analysis
- **Identify** bias patterns in criminal justice ML systems (COMPAS)
- **Analyze** fairness metrics across demographic groups
- **Evaluate** trade-offs between accuracy and fairness

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Algorithmic Fairness** | Ensuring ML models don't discriminate against protected groups |
| **COMPAS** | Controversial criminal risk assessment tool used in US courts |
| **What-If Tool** | Google's interactive tool for ML model analysis and bias detection |
| **Bias Detection** | Methods for identifying unfair model behavior across demographics |
| **Fairness Metrics** | Quantitative measures of model fairness (equality of opportunity, etc.) |
| **Demographic Parity** | Equal positive prediction rates across all groups |
| **Equal Opportunity** | Equal true positive rates across protected groups |

## 🔗 Quick Links

### 📚 Tools & Resources
| Resource | Description | Link |
|----------|-------------|------|
| What-If Tool Tutorial | Interactive census dataset walkthrough | [🔧 Open Tutorial](https://pair-code.github.io/what-if-tool/learn/tutorials/walkthrough/) |
| Census Demo | Web-based What-If Tool demo | [🔧 Open Demo](https://pair-code.github.io/what-if-tool/demos/uci.html) |
| COMPAS Analysis Tool | What-If Tool with COMPAS dataset | [🔧 Open COMPAS Demo](https://pair-code.github.io/what-if-tool/demos/compas.html) |
| Analysis Worksheet | Structured analysis template | [📝 Google Doc](https://docs.google.com/document/d/1XRQTGwyYw6vZ4e2LGDVSM8BEfbxoY16BLYA5OnLF8O8/edit?usp=sharing) |

### 📖 Background Reading
- [Equivant Response to ProPublica](https://web.archive.org/web/20240624143309/https://www.equivant.com/response-to-propublica-demonstrating-accuracy-equity-and-predictive-parity/) - Industry perspective on COMPAS fairness (archived version)

### 🗂️ Navigation
- [← Back to Main Course](../README.md)
- [← Previous: Lab 2 - Data Fundamentals and Intro to ML](../Lab02-Data%20Fundemantal%20and%20Intro%20to%20ML/Lab02-lecture-guide.md)
- [→ Next: Lab 4 - Decision Trees and Fairlearn Evaluation](../Lab04-Decision%20Trees%20and%20Fairlearn%20Evaluation/Lab04-lecture-guide.md)


