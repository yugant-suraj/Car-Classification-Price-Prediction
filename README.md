# 🚗 Car Classification and Price Prediction

> Machine learning system for car classification and used car price prediction based on vehicle attributes.

---

## Overview
This project implements a **machine learning pipeline** to analyze car data and perform:
1. **Car classification** based on features such as brand, model, fuel type, transmission, etc.
2. **Car price prediction** to estimate the market value of a vehicle using historical data.

The goal is to demonstrate **data preprocessing, feature engineering, model training, and evaluation** on a real-world automotive dataset.

---

## Problem Statement
Used car pricing depends on multiple factors including:
- Brand and model
- Manufacturing year
- Mileage
- Fuel type
- Transmission
- Ownership history

Accurately estimating prices manually is error-prone. This project applies **machine learning models** to learn pricing patterns from data and automate predictions.

---

## System Architecture

<pre>
Dataset
  |
  v
Data Cleaning & Preprocessing
  |
  v
Feature Engineering
  |
  v
Train/Test Split
  |
  v
ML Models
  |   \
  |    --> Classification (Car Category / Type)
  |
  ---> Regression (Price Prediction)
  |
  v
Model Evaluation & Prediction
</pre>

---

## Key Features
- Data cleaning and handling of missing values
- Exploratory Data Analysis (EDA)
- Feature encoding for categorical variables
- Supervised learning models for:
  - Car classification
  - Price prediction (regression)
- Model performance evaluation
- Predictive insights from real-world car data

---

## Tech Stack
- Python 3.x
- Pandas
- NumPy
- Matplotlib / Seaborn
- scikit-learn
- Jupyter Notebook

---

## Repository Structure

<pre>
.
├── Major_Project_(V_2_5).ipynb
├── dataset.csv
└── README.md
</pre>

---

## Dataset
The dataset contains structured information about cars, including:
- Brand / Model
- Year of manufacture
- Fuel type
- Transmission
- Mileage
- Selling price

> Note: Dataset may require cleaning and normalization before training.

---

## Models Used
- Linear Regression
- Decision Tree
- Random Forest (if applicable)
- Classification algorithms (based on dataset labels)

---

## Evaluation Metrics
- Regression:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- Classification:
  - Accuracy
  - Confusion Matrix

---

## Usage

1. Open the notebook:
```bash
jupyter notebook Major_Project_(V_2_5).ipynb

