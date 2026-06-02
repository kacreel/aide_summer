# Lab 2.01: Data Fundamentals for Research - Student Guide

## 📋 What You'll Learn Today
- [Overview](#overview)
- [Before We Start](#before-we-start)
- [Session Schedule](#session-schedule)
- [What to Expect](#what-to-expect)
- [Resources](#resources)

## Overview
Welcome to Data Fundamentals! Today you'll learn essential data analysis skills using a real philosophy papers dataset. No prior data science experience required - we'll start with loading data and build up to creating insights and visualizations.

**Duration:** 2 hours  
**What you need:** Basic Python skills from Lab 1  
**Dataset:** Philosophy papers with titles, summaries, and school classifications

## Before We Start

### 📚 What You Should Know
- Basic Python programming (variables, lists, dictionaries, functions)
- Working with strings and file operations
- Understanding of basic programming concepts from Lab 1

### 💻 Technical Setup
Make sure you can access:
- **Google Colab:** [Lab 2.01 Notebook](https://drive.google.com/file/d/1ntYkPcpJj6_VXcnhMhcvKa0hr4BhKPcU/view?usp=sharing)
- **Dataset:** You'll upload `philpapers_enhanced_clean.csv` during the session
- **Libraries:** pandas, matplotlib, numpy (pre-installed in Colab)

### 🎯 Learning Goals
By the end of today, you'll be able to:
- **Load and explore** real research datasets using pandas
- **Filter and search** data to find specific information
- **Clean and transform** data for analysis
- **Create visualizations** to communicate your findings
- **Apply data science** to answer research questions about philosophy

## Session Schedule

### 🚀 Getting Started (5 minutes)
- Quick setup and dataset upload to Colab
- Overview of today's philosophy papers dataset
- Introduction to pandas for data analysis

### 📊 Part 1: Data Loading and First Look (25 minutes)
**What we'll do:** Get familiar with our philosophy papers dataset
- Load data from a CSV file using pandas
- Explore what's in our dataset (how many papers, what information)
- Check data quality and identify any missing information
- Look at paper titles and summaries

**You'll learn:** The first steps of any data analysis project

### 🔍 Part 2: Data Selection & Filtering (25 minutes)
**What we'll do:** Find papers about consciousness (or any topic you're interested in)
- Search for papers containing specific words
- Filter data based on conditions (like paper length)
- Create focused datasets for specific research topics
- Combine multiple search criteria

**You'll learn:** How to extract exactly the information you need from large datasets

### 🛠️ Part 3: Data Cleaning & Feature Engineering (25 minutes)
**What we'll do:** Prepare data for deeper analysis
- Parse complex information (like lists of philosophical schools)
- Create new features (like detecting question titles)
- Categorize papers by summary depth
- Count and analyze philosophical school associations

**You'll learn:** How to transform raw data into analyzable information

### 📈 Part 4: Visualization & Insights (25 minutes)
**What we'll do:** Create charts and discover patterns
- Group data to compare different categories
- Find the most interdisciplinary papers
- Create bar charts to visualize distributions
- Generate insights about research approaches

**You'll learn:** How to communicate findings through visualizations

### 🎉 Summary and Integration (15 minutes)
- Review the complete data analysis workflow you just learned
- Discuss how to apply these skills to your own research
- Q&A and next steps

## What to Expect

### 🛠️ Hands-on Learning
Each part follows the same pattern:
1. **Brief demonstration** - I'll show you the concept (5 min)
2. **Guided practice** - You'll work through 3 tasks with hints (15 min)
3. **Discussion** - We'll review what you discovered (5 min)

### 📊 Real Dataset You'll Work With

**Philosophy Papers Dataset:**
- **What it contains:** Thousands of academic papers from PhilPapers.org
- **Information available:** Titles, abstracts, philosophical schools, text lengths
- **Why it's interesting:** Shows real patterns in philosophical research
- **Your focus:** Papers about consciousness (but you can explore any topic!)

### 🔍 Skills You'll Gain

**Technical Skills:**
- Loading CSV files with `pd.read_csv()`
- Exploring data with `.head()`, `.shape`, `.info()`
- Filtering with boolean conditions and `.str.contains()`
- Creating new columns and categories
- Basic plotting with matplotlib

**Research Skills:**
- Understanding dataset structure and quality
- Asking good questions of your data
- Finding patterns in academic literature
- Communicating findings visually
- Working with real, messy research data

### 📋 What Each Part Covers

| Part | Focus | What You'll Do | Example Output |
|------|-------|----------------|----------------|
| **Part 1** | **Data Exploration** | Load and examine the dataset | "Dataset contains 5,000 papers with 8 features" |
| **Part 2** | **Data Filtering** | Find consciousness papers | "Found 150 papers about consciousness" |
| **Part 3** | **Data Cleaning** | Create new features and categories | "25% of papers ask questions in their titles" |
| **Part 4** | **Visualization** | Make charts and find insights | Bar chart showing summary depth distribution |

## Resources

### 📓 Today's Materials
| Resource | Description |
|----------|-------------|
| [Lab 2.01 Colab Notebook](https://drive.google.com/file/d/1ntYkPcpJj6_VXcnhMhcvKa0hr4BhKPcU/view?usp=sharing) | **Your main workspace** |
| [Local Notebook](Scripts/Lab02.01.Data_Fundamentals_phil.ipynb) | **Backup version** |
| Philosophy Papers Dataset | **Real research data you'll analyze** |

### 📚 Additional Help
- [Pandas Getting Started Guide](https://pandas.pydata.org/docs/getting_started/index.html)
- [Matplotlib Basic Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)

## 🗂️ Navigation
- [← Previous: Lab 1 - Python Fundamentals](../Lab01-Python%20Fundemantals/Lab01-lecture-guide.md)
- [← Back to Main Course](../README.md)
- [→ Next: Lab 2.02 - Introduction to Machine Learning](../Lab02.02-Intro%20to%20Machine%20Learning/Lab02.02-lecture-guide.md)




