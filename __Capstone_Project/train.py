
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import json
import bentoml

df = pd.read_csv('preprocessed_df.csv', header=0)

df['idx'] = df.groupby(['playlist_id']).ngroup()
res = dict(zip(df.idx, df.playlist_name))

#print(res)
#[print(key,':',value) for key, value in res.items()]

with open("playlists_name.json", "w") as fp:
    json.dump(res, fp)


y_prep = df.idx.values

df_prep = df.drop(columns=['track_title', 'track_artist', 'playlist_id','playlist_name', 'id', 'idx'])
df_prep['now']= pd.to_datetime(pd.datetime.now().date())
df_prep['release_date']= pd.to_datetime(df_prep['release_date'])
df_prep['counting_days'] = (df_prep['now'] - df_prep['release_date']).dt.days

df_prep = df_prep.drop(columns=['release_date', 'now'])
# we will need to preprocess this even when we'll receive the input song to evaluate.


df_train, df_test, y_train, y_test = train_test_split(df_prep, y_prep, test_size=0.2, random_state=1, stratify=y_prep)
#stratify is necessarty in order to keep a proportional target class distribution in the train/test split

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
    'eta': 0.8, 
    'learning_rate': 0.1, 
    'objective': 'multi:softprob',
    'num_class': 50,
    'max_depth': 25,
    'n_estimators': 50,
    'nthread': 10,
    'colsample_bytree': 0.5,
    'subsample' : 0.5,
    'eval_metric': 'merror',
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  verbose_eval=5, evals=watchlist)


bentoml.xgboost.save_model(
    'spotify_playlist_matcher',
    model,
    custom_objects={
        'dictVectorizer': dv,
        
    }
)

