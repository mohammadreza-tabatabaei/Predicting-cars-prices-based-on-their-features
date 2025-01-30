# Requirements:
###### 1.Google Colab
###### 2.Pandas
###### 3.sklearn
###### 4.Numpy
###### 5.matplotlib
###### 6.seaborn


# Dataset:
The dataset provided by AutoTrader is to be used only by students enrolled on 6G7V0015 (24/25). This dataset contains the prices of 400,000 cars, and for each car, characteristics such as year of manufacture, manufacturer, mileage, car color, fuel type, and body type.

# Project Overview:
This project focuses on predicting car prices using machine learning techniques. The dataset consists of car listings with various features such as mileage, fuel type, manufacturing year, brand, and vehicle condition. Multiple data preprocessing steps and machine learning models were applied to achieve accurate predictions.

In this project, the goal was to predict the price of a car based on its features. I used many techniques in this project, including Target Encoding, lable encoding,scaling, Cross-Validation, LinearRegression,KNeighborsRegressor, DecisionTreeRegressor, Grid Search, Feature Importance, Instance-Level Errors, Top Errors

# Techniques Used:
## 1. Data Preprocessing & Cleaning
### Handling Missing Values:

Numerical features were filled using median imputation with SimpleImputer.
Categorical features were replaced with 'Unknown' to avoid data loss.
### Feature Encoding:
One-Hot Encoding: Applied to the top 5 most frequent car brands to group rare brands into an "Other" category.
Target Encoding: Applied to car color using mean price mapping.
Label Encoding: Used for fuel type, body type, and vehicle condition.
## 2.Feature Scaling & Transformation
Power Transformer was used to normalize numerical features for better model performance.
## 3. Exploratory Data Analysis (EDA)
Visualizations:
Histograms & Violin Plots: Distribution of car prices before and after transformation.
Scatter Plots: Relationship between price and mileage, year, fuel type, and brand.
Heatmaps: Correlation between numerical features.
## 4. Model Training & Evaluation
### Models Used:
Linear Regression
k-Nearest Neighbors (k-NN)
Decision Tree Regressor
### Hyperparameter Tuning:
Grid Search with Cross-Validation was used to find the best parameters for each model.
### Evaluation Metrics:
Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R² Score
## 5. Model Interpretation
### Feature Importance Analysis:

Decision Tree was used to extract feature importance.
Grouped importance was calculated for car brands to reduce dimensionality.
### Instance-Level Error Analysis:

The top 10 highest and lowest prediction errors were examined to identify model weaknesses.

## Results & Insights
k-NN achieved the best performance with the highest R² score.
Power Transformation significantly improved model performance.
Feature selection and encoding had a large impact on prediction accuracy.

## Future Improvements
Experiment with ensemble models like Random Forest & XGBoost.
Incorporate additional features like location and seller type.
Deploy the model using Flask or FastAPI.

This phase focuses on recognizing tomato pests. The recognition phase makes use of SVM, decision tree, and KNN algorithm classifires.
