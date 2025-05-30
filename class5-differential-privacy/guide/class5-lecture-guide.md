# Class 5: Differential Privacy - Lecture Guide

## Overview
Introduction to differential privacy concepts, de-identification attacks, and privacy-preserving machine learning techniques.

## Pre-Class Reading
### Required Materials:
1. **Programming Differential Privacy Chapter 1**
   - [Chapter 1: Programming DP](https://programming-dp.com/ch1.html)
   - Focus on core concepts and motivation

## Class Schedule

### Session 1: De-identification Attacks (30 minutes)
**Understanding Privacy Vulnerabilities**
- **Exercise:** [De-identification Attack Lab](https://replit.com/@allenol/De-identification)
- **Solutions:** [De-identification Solutions](https://replit.com/@allenol/De-Identification-Solutions)
- Demonstrate how seemingly anonymous data can be re-identified
- Real-world examples of privacy breaches

### Session 2: Differential Privacy Fundamentals (25 minutes)
**Core Concepts and Theory**
- Definition of differential privacy
- Epsilon and delta parameters
- Privacy-utility trade-offs
- Mathematical foundations (accessible level)

### Session 3: Simple Differential Privacy Example (30 minutes)
**Hands-on Implementation**
- **Tutorial:** [Differential Privacy for Beginners](https://towardsdatascience.com/a-differential-privacy-example-for-beginners-ef3c23f69401)
- Implement basic noise addition mechanisms
- Compare results with and without privacy protection
- Understand the impact of different privacy parameters

### Session 4: Privacy-Preserving ML (20 minutes)
**Applications in Machine Learning**
- How differential privacy applies to ML models
- Training with privacy guarantees
- Evaluation of privacy-preserving algorithms

### Discussion and Extensions (10 minutes)
- **TODO:** Discuss potential extensions to curriculum
- Explore advanced topics for interested students
- Connect to current privacy regulations (GDPR, CCPA)

## Learning Objectives
By the end of this class, students should be able to:
- Understand the limitations of traditional anonymization
- Explain the concept of differential privacy
- Implement basic differential privacy mechanisms
- Recognize privacy vulnerabilities in data sharing
- Apply privacy-preserving techniques to machine learning

## Key Concepts
- **De-identification:** Removing obvious identifiers from data
- **Re-identification:** Process of linking anonymous data back to individuals
- **Differential Privacy:** Mathematical framework for privacy protection
- **Epsilon (ε):** Privacy budget parameter
- **Noise Addition:** Core mechanism for achieving differential privacy

## Materials Needed
- Computer with internet access
- Replit account
- Python environment with numpy/pandas
- Access to tutorial materials
