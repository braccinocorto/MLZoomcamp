import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

df = pd.read_csv('healthcare-dataset-stroke-data.csv', header=0)
df['bmi'].fillna(df['bmi'].mean(),inplace=True)

df = df.drop(columns='id')

y_prep = df.stroke
df_prep = df.drop(columns='stroke')

#build train and validation set with sklearn
# random state for reproducibility of the test.

df_train, df_test = train_test_split(df_prep, test_size=0.2, random_state=42)

y_train, y_test = train_test_split(y_prep, test_size=0.2, random_state=42)

train_dict = df_train.to_dict(orient='records')
test_dict = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
X_test = dv.transform(test_dict)

features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dtrain, 'train'), (dtest, 'test')]

xgb_params = {
    'eta': 0.05, 
    'max_depth': 2,
    'min_child_weight': 10,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    'eval_metric': 'auc',
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                  verbose_eval=5, evals=watchlist)


import bentoml
bentoml.xgboost.save_model(
    'heart_booster_tree',
    model,
    custom_objects={
        'dictVectorizer': dv
    })


#this is a test to see how it behaves the model 

#import json
#request = df_test.iloc[0].to_dict()
#print(json.dumps(request, indent=2))

