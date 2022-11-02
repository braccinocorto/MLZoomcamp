# Heart Stroke prediction

## Problem description
Heart stroke looks like unpredictable. We try to analyze this dataset and see if we can have a prediction based on some feature analysis.
The model intends to predict if, given a set of features, what's the probability for that individual to get stroke.

This is a binary classification project.

The dataset used to train the model is taken from kaggle:
url: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv
[download][linkdata]

## EDA
- Classification of the features
- Distribution of values of the features
- Correlation Matrix on the raw data, including categorical values


## Model Training
- Split the dataset in 80/20/20 train/validation/test
- Train a Logistic Regressor
- Train a Randomforest
- Train XGBoost
- Evaluation of the best performing model [via AUC_score]
- Selection of the model

## Exporting the notebook
- With the selected model, create the train.py file. For training data, it uses local csv file (the same as downloaded).
-

## Environment

## Deployment




   [linkdata]: <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv>