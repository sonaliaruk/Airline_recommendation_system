# Indigo_Airline_recommendation_system
ML project to predict airline recommendation using Streamlit

![Airline Image](https://github.com/sonaliaruk/Airline_recommendation_system/blob/main/airplane_image.jpg?raw=true)

This project focuses on analyzing and classifying airline passenger reviews using machine learning models. The goal is to predict whether a passenger would recommend the airline based on various review features, and extract actionable insights that airlines can use to improve customer satisfaction.

# Dataset Overview 

* Source: Airline reviews from 2006 to 2019
* Features: 17 columns including ratings, cabin class, traveler type, etc.
* Target Variable: recommended (Yes/No)

# 🚀 Project Workflow

Data Preparation
* Transformed date-related columns into proper datetime format
* Standardized rating values for consistency
* Handled missing and inconsistent data through cleaning techniques

Exploratory Data Analysis (EDA)
* Analyzed airline popularity and customer preferences
* Studied trends across cabin classes and service ratings
* Visualized insights using bar charts, pie charts, and trend plots

Statistical Analysis
* Conducted hypothesis testing using:
- T-Test
- Chi-Square Test
- ANOVA
Examined relationships between traveler type, cabin class, and satisfaction levels

Feature Engineering
* Converted categorical variables into numerical form using encoding techniques
* Considered dimensionality reduction, but skipped due to already strong model performance

Model Development
* Implemented multiple machine learning models:
Decision Tree , Random Forest , XGBoost
* Dataset split into:
70% Training
30% Testing

Streamlit Web Application
* Built an interactive user interface using Streamlit
* Enabled users to input passenger experience ratings
- Displayed:
* Recommendation result (Yes/No)
* Probability score (%)
- Developed a mini dashboard showing:
* Overall recommendation rate
* Average service ratings
* Weakest performance areas
