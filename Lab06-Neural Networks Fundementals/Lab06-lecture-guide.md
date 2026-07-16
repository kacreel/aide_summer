# Lab 6: Neural Networks Fundamentals - Student Guide

## 📋 What You'll Learn Today
- [Overview](#overview)
- [Before We Start](#before-we-start)
- [Session Schedule](#session-schedule)
- [What to Expect](#what-to-expect)
- [Resources](#resources)

## Overview
Welcome to Neural Networks Fundamentals! Today you'll learn the mathematical foundations of deep learning by working through calculations by hand. No coding required - we'll focus on understanding how neural networks think and learn at the most basic level.

**Duration:** 2 hours  
**What you need:** Basic algebra and willingness to work through math step-by-step  
**Approach:** Hand calculations and visual understanding

## Before We Start

### 📚 What You Should Know
- Basic algebra and arithmetic
- Understanding of functions (input → output)
- Familiarity with graphs and coordinates
- Matrix multiplication basics (we'll review this)


### 🎯 Learning Goals
By the end of today, you'll be able to:
- **Calculate perceptron outputs** by hand for simple decisions
- **Understand neural network architecture** and how layers connect
- **Perform forward propagation** step-by-step through a network
- **Grasp backpropagation basics** and how networks learn from mistakes
- **Visualize how networks** process information
- **Prepare for advanced topics** like transformers and embeddings

## Session Schedule

### 🚀 Getting Started (10 minutes)
- Quick review: What are neural networks and why do they work?
- Overview of today's two main exercises
- Connection to biological neurons and decision-making

### 🎯 Session 1: Perceptron by Hand (50 minutes)
**What we'll do:** Build your first "artificial brain cell"
- **Exercise**: Work through the "Should we go to the beach?" decision problem
- Calculate weighted inputs step-by-step
- Apply activation functions to make binary decisions
- Understand the geometry of linear decision boundaries
- Practice with multiple examples and variations
- Explore different scenarios and weight adjustments

**You'll learn:** How the simplest neural network makes decisions

**Materials**: [Perceptron Practice Exercise](./Exercises/01.PERCEPTRONs.pdf)

### Break (10 minutes)

### 🔄 Session 2: Backpropagation by Hand (50 minutes)
**What we'll do:** Discover how networks learn from their mistakes
- Start with a simple multi-layer network example
- Make a prediction and calculate the error
- Work backwards through the network using chain rule
- Calculate gradients and weight updates step-by-step
- Understand gradient descent intuitively
- Practice with a complete learning cycle example

**You'll learn:** The magic behind how neural networks improve over time

**Materials**: [Neural Networks Tutorial with Backpropagation Examples](./Exercises/02.NeuralNetworks.pdf)

### 🎉 Wrap-up and Reflection (10 minutes)
- Review the complete journey from simple decisions to learning
- Discuss what surprised you most about the mathematics
- Connect to modern AI: How these principles scale to large networks
- Preview Lab 7: Transformers and Embeddings

## What to Expect

### 🛠️ Hands-on Mathematics
This lab is **calculation-intensive** but **beginner-friendly**:
- **Step-by-step guidance**: Every calculation broken down clearly
- **Real examples**: "Should we go to the beach?" and other relatable decisions
- **Visual aids**: Diagrams showing information flow
- **Immediate understanding**: See exactly how each step contributes

### 📊 The Problems You'll Solve

**Perceptron Example**: "Should we go to the beach?"
- **Inputs**: Weather conditions (sunny, temperature, humidity)
- **Process**: Weight each factor and make a decision
- **Output**: Yes/No decision with confidence

**Neural Network Example**: Complex pattern recognition
- **Architecture**: Input → Hidden Layer → Output
- **Forward Pass**: Calculate predictions
- **Backward Pass**: Learn from mistakes

### 🔍 Skills You'll Develop

**Mathematical Skills**:
- Matrix multiplication in neural network context
- Understanding of dot products and weighted sums
- Function composition and chain rule basics
- Gradient calculation intuition

**Conceptual Skills**:
- Neural network architecture design
- Information flow in deep networks
- Learning algorithm mechanics
- Connection between math and AI behavior

### 📋 Session Structure

| Session | Focus | Method | Key Insight |
|---------|-------|--------|-------------|
| **Session 1** | **Perceptron** | Hand calculations with real example | Simple decisions through weighted voting |
| **Session 2** | **Backpropagation** | Learning by working backwards | How networks improve from mistakes |

## Resources

### 📓 Today's Materials
| Resource | Description |
|----------|-------------|
| [Perceptron Exercise](./Exercises/01.PERCEPTRONs.pdf) | **Step-by-step practice problems** |
| [Neural Networks Tutorial](./Exercises/02.NeuralNetworks.pdf) | **Worked examples with complete calculations** |

### 📚 Background Study 
| Resource | Description |
|----------|-------------|
| [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | **Visual explanations (watch before or after)** |
| [Matrix Multiplication Guide](https://www.mathsisfun.com/algebra/matrix-multiplying.html) | **Math refresher if needed** |
| [W3Schools Perceptrons](https://www.w3schools.com/ai/ai_perceptrons.asp) | **Additional perceptron examples** |


### 🔬 Advanced Topics to Explore Later (Optional)
- **Convolutional Neural Networks (CNNs)**: [Detailed CNN Tutorial](./Tutorials/03.Convolutional%20Neural%20Networks(CNNs).md) - Learn how neural networks process images


## 🗂️ Navigation
- [← Previous: Lab 5 - Differential Privacy](../Lab05-Differential%20Privacy/Lab05-lecture-guide.md)
- [← Back to Main Course](../README.md)
- [→ Next: Lab 7 - Transformers and Embeddings Fundamentals](../Lab07-Transformers%20and%20Embeddings%20Fundementals/Lab07-lecture-guide.md)