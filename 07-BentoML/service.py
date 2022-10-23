import numpy as np

import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
#dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_hmwk", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    #vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(input_series)
    #print(prediction)
    return prediction

#    result = prediction[0]

#    if result > 0.5:
#        return {
#            "status": "DECLINED"
#        }
#    elif result > 0.25:
#        return {
#            "status": "MAYBE"
#        }
#    else:
#        return {
#            "status": "APPROVED"
#        }
