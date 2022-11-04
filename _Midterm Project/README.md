# Heart Stroke prediction

## Problem description
Heart stroke looks like unpredictable. We try to analyze this dataset and see if we can have a prediction based on some feature analysis.
The model intends to predict if, given a set of features, what's the probability for that individual to get stroke.

This is a binary classification project.

The dataset used to train the model is taken from kaggle:
url: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv
[download][linkdata]

The target value is the column "stroke", which is binary.

## EDA
- Classification of the features
- Distribution of values of the features
- Correlation Matrix on the raw data, including categorical values


## Model Training
- Split the dataset in 80/20/20 train/validation/test
- The training process aims to compare the AUC value for the regressors analyzed. So that we can compare them on the same field.
- Train a Logistic Regressor. I've tested the different regression algos with different C values. 
- Train a Randomforest. I've explored the combinations of n_estimators and max_depth.
- Train XGBoost. To find the best combination, through capture output (with an iteratos number of 500, we test differentv  values of eta, max depth, min child weight)
- Evaluation of the best performing model [via AUC_score confrontation]
- Selection of the model: XGBoost.

## Exporting the notebook
- With the selected model, and hyperparameter tuning, I created the train.py file. For training data, it uses local csv file (the same as downloaded).
- Created a test.py so that we can load the model (via Bento), provide data (in JSON format) and receive a response
- Created a predict.py file, based on BentoML, ready for the deployment. Included a pydantic model in order to prevent erroneous data.
- The output received from the model is a probability, and as such is treated (with a threshold of 0.5), even if some other level of risk are provided.
- Pydantic requirements to be added with the @validator decorators

## Environment
- BentoML

## Deployment




   [linkdata]: <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv>