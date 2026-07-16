# Lab 5: Differential Privacy - Lecture Guide

## 📋 Table of Contents
- [Overview](#overview)
- [Pre-Lab Learning](#pre-lab-learning)
- [Lab Schedule](#lab-schedule)
- [Learning Objectives](#learning-objectives)
- [Key Concepts](#key-concepts)
- [Quick Links](#quick-links)

## Overview
Introduction to differential privacy concepts, de-identification attacks, and privacy-preserving machine learning techniques. Students will learn why traditional anonymization fails and how differential privacy provides mathematical guarantees for privacy protection while maintaining data utility.

## Pre-Lab Learning (45 minutes)
### Required Materials:
1. **Core Reading** (30 minutes)
   - [Programming Differential Privacy - Chapter 1](https://programming-dp.com/chapter1.html) - Fundamental concepts and mathematical framework
   
2. **Supplementary Reading** (15 minutes)
   - [Differential Privacy for Beginners](https://towardsdatascience.com/a-differential-privacy-example-for-beginners-ef3c23f69401) - Practical introduction with examples

## Lab Schedule

### Session 1: Differential Privacy Fundamentals (30 minutes)
**Core Concepts and Theory**
- **Tutorial:** `Census/census_tutorial.ipynb` | [📁 Local File](./Scripts/Census/census_tutorial.ipynb) | [☁️ Google Colab](https://colab.research.google.com/drive/1h0ZOYBOzeEAWhKg2nvDtgRFO9CZr4ddy?usp=sharing)
- Based on [Programming DP Chapter 1](https://programming-dp.com/ch1.html)
- De-identification
- Re-identification
- Identifying information / personally identifying information
- Linkage attacks
- Aggregation and aggregate statistics
- Differencing attacks

### Session 2: Census Dataset Exercise (30 minutes)
**De-identification Attacks and Basic DP**
- **Exercise:** `Census/census_exercises.ipynb` | [📁 Local File](./Scripts/Census/census_exercises.ipynb) | [☁️ Google Colab](https://colab.research.google.com/drive/15e682dGaOVC_uwSvSJaqfMR9kOFQFBys?usp=sharing)
- Demonstrate re-identification attacks on census data
- Implement basic differential privacy mechanisms
- Compare results with and without privacy protection

### Session 3: Sleep Dataset Exercise (30 minutes)
**Practical Differential Privacy Implementation**
- **Exercise:** `Sleep/differentialPrivacy_exercises.ipynb` | [📁 Local File](./Scripts/Sleep/differentialPrivacy_exercises.ipynb) | [☁️ Google Colab](https://colab.research.google.com/drive/1xQI3AvuB6UopswqcC0HKiFfHrV7GavzB?usp=sharing)
- Based on [Differential Privacy for Beginners](https://towardsdatascience.com/a-differential-privacy-example-for-beginners-ef3c23f69401)
- Hands-on noise addition mechanisms
- Understand the impact of different privacy parameters
- Privacy-preserving data analysis

## Learning Objectives
By the end of this lab, students should be able to:
- **Understand Privacy Attacks**: Perform linkage and differencing attacks on seemingly anonymized data
- **Apply Differential Privacy**: Implement basic differential privacy mechanisms using noise addition
- **Evaluate Privacy-Utility Trade-offs**: Balance privacy protection with data utility for practical applications
- **Use Privacy Tools**: Work with PyDP library and understand privacy parameters (epsilon, delta)
- **Recognize Privacy Vulnerabilities**: Identify when traditional anonymization methods fail

## Key Concepts

### 🔍 Privacy Attack Techniques
- **Linkage Attacks**: Using auxiliary information to re-identify individuals
- **Differencing Attacks**: Exploiting differences in aggregate statistics
- **Aggregation Vulnerabilities**: How even aggregated data can leak individual information
- **De-identification Limitations**: Why removing direct identifiers isn't sufficient

### 🛡️ Differential Privacy Fundamentals  
- **Mathematical Framework**: Formal definition of differential privacy
- **Privacy Parameters**: Understanding epsilon (ε) and delta (δ) values
- **Privacy-Utility Trade-off**: Balancing privacy protection with data usefulness
- **Noise Addition**: Laplacian and Gaussian noise mechanisms

### 🔧 Practical Implementation
- **PyDP Library**: Google's differential privacy library for Python
- **Privacy Functions**: BoundedSum, BoundedMean, Count, and Max operations
- **Randomized Response**: Classic technique for sensitive survey data
- **Privacy Budgeting**: Managing cumulative privacy loss across queries

### 📊 Real-world Applications
- **Regulatory Compliance**: GDPR, CCPA, and privacy legislation requirements
- **Industry Use Cases**: Census data, healthcare records, location data
- **Privacy-Preserving Analytics**: Statistical analysis without compromising privacy

## 🔗 Quick Links

### 📚 Exercise Files
| Exercise | Local File | Google Colab |
|----------|------------|--------------|
| Census Tutorial | [📁 census_tutorial.ipynb](./Scripts/Census/census_tutorial.ipynb) | [☁️ Open in Colab](https://colab.research.google.com/drive/1h0ZOYBOzeEAWhKg2nvDtgRFO9CZr4ddy?usp=sharing) |
| Census Exercises | [📁 census_exercises.ipynb](./Scripts/Census/census_exercises.ipynb) | [☁️ Open in Colab](https://colab.research.google.com/drive/15e682dGaOVC_uwSvSJaqfMR9kOFQFBys?usp=sharing) |
| Sleep Exercises | [📁 differentialPrivacy_exercises.ipynb](./Scripts/Sleep/differentialPrivacy_exercises.ipynb) | [☁️ Open in Colab](https://colab.research.google.com/drive/1xQI3AvuB6UopswqcC0HKiFfHrV7GavzB?usp=sharing) |

### 📖 Reference Materials
- [Programming Differential Privacy - Chapter 1](https://programming-dp.com/ch1.html)
- [Differential Privacy for Beginners](https://towardsdatascience.com/a-differential-privacy-example-for-beginners-ef3c23f69401)

### 🗂️ Navigation
- [← Back to Main Course](../README.md)
- [← Previous: Lab 4 - Advanced Decision Trees and Fairlearn Evaluation](../Lab04-Decision%20Trees%20and%20Fairlearn%20Evaluation/Lab04-lecture-guide.md)
- [→ Next: Lab 6 - Neural Networks Fundamentals](../Lab06-Neural%20Networks%20Fundementals/Lab06-lecture-guide.md)







