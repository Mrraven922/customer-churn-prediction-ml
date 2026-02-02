# customer-churn-prediction-ml
Machine learning-based customer churn prediction application with real-time inference deployment.

## Overview

This repository presents an end-to-end machine learning solution for predicting customer churn in subscription-based and financial service environments.

The project demonstrates a production-oriented data science workflow, including data preprocessing, feature engineering, supervised model training, artifact serialization, and deployment through a lightweight inference application.

The solution is designed with enterprise analytics use cases in mind, emphasizing reproducibility, scalability, and deployment readiness.

## Business Context

Customer churn directly impacts revenue, profitability, and long-term customer lifetime value.

Financial institutions and subscription-based platforms process millions of customers daily. Even a small churn percentage can result in significant financial loss and operational inefficiencies.

Accurate churn prediction enables organizations to:

Identify high-risk customers early

Optimize retention campaigns

Improve customer engagement

Reduce revenue leakage

Support data-driven business strategies

## Objective

To develop a machine learning-powered churn prediction system with a focus on:

Reliable binary classification performance

Production-ready preprocessing pipelines

Consistent inference behavior

Business-aligned output interpretation

Deployment simulation using real-time UI

## Technical Approach
Problem Type

Binary Classification

Machine Learning Framework

Scikit-learn

Model Type

Supervised classification model

Preprocessing

The preprocessing workflow includes:

Categorical variable encoding (Gender)

Feature scaling using StandardScaler

Numerical normalization

Feature ordering consistency

These steps ensure stable training performance and reliable real-time inference.

## Pipeline Design

The project follows a modular and production-aligned design:

Training and experimentation handled separately

Preprocessing artifacts persisted for reuse

Inference pipeline decoupled from training logic

This structure reflects enterprise ML system design principles.

## Model Persistence

Trained components are serialized using Joblib:

model.pkl — Production inference model

scaler.pkl — Feature transformation pipeline

This guarantees consistency between training and deployment environments.

## System Architecture
Raw Customer Data
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Preprocessing Pipeline
        ↓
Model Training & Evaluation
        ↓
Serialized Model Artifacts
        ↓
Streamlit Inference Application

Application Output (Inference UI)

The project includes a Streamlit-based inference application that allows users to:

Enter customer attributes manually

Perform real-time churn predictions

View clear binary classification results

This interface simulates an internal business analytics tool used by retention and operations teams.

## Project Structure
customer-churn-prediction-ml/


├── customer_churn_data.csv      # Dataset

├── churn_prediction.ipynb       # EDA and model training


├── app.py                       # Streamlit inference application

├── model.pkl                    # Trained ML model

├── scaler.pkl                   # Feature scaler

├── README.md                    # Project documentation


## Dataset Summary

Each customer record contains:

Age

Gender

Tenure

Monthly charges

These features are commonly used in churn modeling across enterprise analytics systems.

## Target Variable
Churn

1 → Customer likely to churn

0 → Customer retained

Model Evaluation Considerations

Model performance is evaluated with business relevance in mind.

## Risk Interpretation

False Negatives (Missed Churn)
→ Direct revenue loss and missed retention opportunity

False Positives (Incorrect Churn Flag)
→ Increased operational cost and potential customer dissatisfaction

## Deployment

A lightweight Streamlit application demonstrates:

Integration of ML models into real applications

Real-time inference capabilities

Production-style deployment workflow

## Application Output (Inference UI)
<img width="1885" height="958" alt="output" src="https://github.com/user-attachments/assets/2dafc7a8-89ce-48d5-ab78-f8d03488ab0b" />



## Setup Instructions
Clone Repository
git clone https://github.com/your-username/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml

## Install Dependencies
pip install pandas numpy scikit-learn streamlit joblib

## Run Application
streamlit run app.py

## Engineering Best Practices Demonstrated

End-to-end reproducibility

Pipeline-based preprocessing

Separation of training and inference logic

Model artifact versioning

Enterprise-style project organization

Deployment simulation

Potential Enhancements

Probability score output

Model explainability (SHAP)

REST API integration (FastAPI)

Batch prediction support

Docker containerization

Cloud deployment

Monitoring and drift detection

CI/CD integration

Author

Your Name : Vignesh Raj
