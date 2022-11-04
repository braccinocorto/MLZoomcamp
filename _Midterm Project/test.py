import bentoml
import xgboost as xgb
import numpy as np


test_dict =  {
  "gender": "Male",
  "age": 80.0,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Rural",
  "avg_glucose_level": 105.92,
  "bmi": 32.5,
  "smoking_status": "never smoked"
}


booster = bentoml.xgboost.load_model("heart_booster_tree:latest")

model_ref = bentoml.xgboost.get("heart_booster_tree:latest")
dv = model_ref.custom_objects['dictVectorizer']

DMready = dv.transform(test_dict)

pred = booster.predict(xgb.DMatrix( 
   DMready)
)

print(pred)
