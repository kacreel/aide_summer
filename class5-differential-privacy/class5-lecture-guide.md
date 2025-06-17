# Class 5: Differential Privacy - Lecture Guide

## 📋 Table of Contents
- [Class Schedule](#class-schedule)
- [Learning Objectives & Key Concepts](#learning-objectives--key-concepts)
- [Quick Links](#quick-links)

## Overview
Introduction to differential privacy concepts, de-identification attacks, and privacy-preserving machine learning techniques.

## Class Schedule

### Session 1: Differential Privacy Fundamentals (20 minutes)
**Core Concepts and Theory**
- Definition of differential privacy
- Epsilon and delta parameters
- Privacy-utility trade-offs
- Mathematical foundations (accessible level)

### Session 2: Census Dataset Exercise (35 minutes)
**De-identification Attacks and Basic DP**
- **Exercise:** `Census/census_exercises.ipynb` | [📁 Local File](./Exercises/Census/census_exercises.ipynb) | [☁️ Google Colab](https://colab.research.google.com/drive/15e682dGaOVC_uwSvSJaqfMR9kOFQFBys?usp=sharing)
- **Tutorial:** `Census/census_tutorial.ipynb` | [📁 Local File](./Exercises/Census/census_tutorial.ipynb) | [☁️ Google Colab](https://colab.research.google.com/drive/1h0ZOYBOzeEAWhKg2nvDtgRFO9CZr4ddy?usp=sharing)
- Based on [Programming DP Chapter 1](https://programming-dp.com/ch1.html)
- Demonstrate re-identification attacks on census data
- Implement basic differential privacy mechanisms
- Compare results with and without privacy protection

### Session 3: Sleep Dataset Exercise (30 minutes)
**Practical Differential Privacy Implementation**
- **Exercise:** `Sleep/differentialPrivacy_exercises.ipynb` | [📁 Local File](./Exercises/Sleep/differentialPrivacy_exercises.ipynb) | [☁️ Google Colab](https://colab.research.google.com/drive/1xQI3AvuB6UopswqcC0HKiFfHrV7GavzB?usp=sharing)
- Based on [Differential Privacy for Beginners](https://towardsdatascience.com/a-differential-privacy-example-for-beginners-ef3c23f69401)
- Hands-on noise addition mechanisms
- Understand the impact of different privacy parameters
- Privacy-preserving data analysis


## Learning Objectives & Key Concepts
By the end of this class, students should be able to:
- Understand **de-identification** limitations and **re-identification** attacks
- Explain **differential privacy** as a mathematical framework for privacy protection
- Implement basic differential privacy mechanisms using **noise addition**
- Work with **epsilon (ε)** privacy budget parameters
- Apply privacy-preserving techniques to machine learning datasets

## 🔗 Quick Links

### 📚 Exercise Files
| Exercise | Local File | Google Colab |
|----------|------------|--------------|
| Census Tutorial | [📁 census_tutorial.ipynb](./Exercises/Census/census_tutorial.ipynb) | [☁️ Open in Colab](https://colab.research.google.com/drive/1h0ZOYBOzeEAWhKg2nvDtgRFO9CZr4ddy?usp=sharing) |
| Census Exercises | [📁 census_exercises.ipynb](./Exercises/Census/census_exercises.ipynb) | [☁️ Open in Colab](https://colab.research.google.com/drive/15e682dGaOVC_uwSvSJaqfMR9kOFQFBys?usp=sharing) |
| Sleep Exercises | [📁 differentialPrivacy_exercises.ipynb](./Exercises/Sleep/differentialPrivacy_exercises.ipynb) | [☁️ Open in Colab](https://colab.research.google.com/drive/1xQI3AvuB6UopswqcC0HKiFfHrV7GavzB?usp=sharing) |






