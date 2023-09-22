# telco-customer-churn

Telco Customer Churn Data Analysis Project
Overview
This repository contains a data analysis project focused on understanding customer churn in the telecommunications (telco) industry. The project utilizes a dataset related to telco customer behavior and aims to identify patterns and factors that contribute to customer churn. The findings and insights obtained from this analysis can assist telco companies in implementing strategies to reduce customer attrition and improve customer retention.

Dataset
The dataset used for this analysis is included in the repository as a CSV file named telco_customer_churn.csv. It contains relevant features and customer churn labels necessary for conducting the analysis.

Analysis
The main analysis of the telco customer churn data is documented in a Jupyter notebook named Telcomchurn.ipynb. The notebook outlines the steps taken for data exploration, preprocessing, feature engineering, model selection, evaluation, and interpretation of results.

Dependencies
The analysis uses the following Python libraries:

-Pandas
-NumPy
-Matplotlib
-Seaborn
-Scikit-Learn

Machine Lerning Models-
- NaiveBayes
- Logistic Regression 
-Random Forest
-XGBoost

SMOTE TECHNIQUE-
-SMOTE for Random Forest
-SMOTE for Logistic Regression
-SMOTE for XGBoost
-SMOTE for Naive Bayes

Hyperparameter Tuning Of Models-
-Logistic Regression
-XGboost

According to the results,out of both stages of analysis the most accurate model is Naive Bayes with an accuracy of 88%.


To be noted that naive Bayes and Xgboost has similar recall results after applying SMOTE as well for Logistic regression and Xgboost.

Out of the both the Hyperparameter tuned models,random Forest has a higher recall value with 0.79.
