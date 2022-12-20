import bentoml
import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import date
import json

with open("playlists_name.json", "r") as fp:
  res = json.load(fp)

print(res)
print(type(res))

test_dict =  {

"danceability":0.793,
"energy":0.698,
"key":11,
"loudness":-3.626,
"mode":1,
"speechiness":0.104,
"acousticness":0.163,
"instrumentalness":0.145,
"liveness":0.0745,
"valence":0.339,
"tempo":130.0,
"duration_ms":285571,
"time_signature":4,
"popularity":65,
"release_date":"2003-05-23"
}


booster = bentoml.xgboost.load_model("spotify_playlist_matcher:latest")

model_ref = bentoml.xgboost.get("spotify_playlist_matcher:latest")
dv = model_ref.custom_objects['dictVectorizer']



#count_days = (pd.to_datetime(pd.datetime.now().date()) - pd.to_datetime(test_dict["release_date"])).days

count_days = (pd.to_datetime(date.today()) - pd.to_datetime(test_dict["release_date"])).days

print(count_days)
test_dict["counting_days"] = count_days
test_dict.pop("release_date")

DMready = dv.transform(test_dict)
dict_test = xgb.DMatrix(DMready)
pred_prob = booster.predict(dict_test, output_margin=True)
pred_label = np.argmax(pred_prob,axis=1)

print("Pred class:")
print(pred_prob)

print("Pred label:")
print(pred_label)
print(res[str(pred_label.item())])


#test songs:
# see the file "sample songs.txt"
