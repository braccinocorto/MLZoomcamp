import numpy as np

import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("heart_booster_tree:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("heart_stroke_model", runners=[model_runner])

from pydantic import BaseModel

class HeartStrokeApp(BaseModel):
    gender: str 
    age: float
    hypertension: int  
    heart_disease: int  
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@svc.api(input=JSON(pydantic_model= HeartStrokeApp), output=JSON())
async def classify_heartstroke(heart_stroke_pred):
    application_data = heart_stroke_pred.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    result = prediction[0]
    if result > 0.5:
        return {
            "response": "HEARTSTROKE LIKELY"
        }
    elif result > 0.20:
        return {
            "response": "POSSIBLE HEARTSTROKE"
        }
    else:
        return {
            "response": "HEARTSTROKE NOT LIKELY"
        }


# result is the probability of heartstroke
# based on the dataset, a value > than 0.5 provides a likely heartstroke outcome
# 0.20 is the threshold where the model predictions have a probability distribution that of the training data on the dataset. 
# So, a value above the threshold may lead to a condition which is 'above the average' (based on the dataset)

