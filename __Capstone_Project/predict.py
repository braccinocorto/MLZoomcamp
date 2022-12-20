import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON
import json

model_ref = bentoml.xgboost.get("spotify_playlist_matcher:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("spotify_playlist_model", runners=[model_runner])

from pydantic import BaseModel

class SpotifyPlaylistApp(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float 
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float 
    tempo :float
    duration_ms: int
    time_signature: int
    popularity: int
    counting_days: int


@svc.api(input=JSON(pydantic_model= SpotifyPlaylistApp), output=JSON())
async def classify_playlist(track_to_classify):
    with open("playlists_name.json", "r") as fp:
        res = json.load(fp)

    application_data = track_to_classify.dict()
    vector = dv.transform(application_data)
    pred_prob = await model_runner.predict.async_run(vector)
    pred_label = np.argmax(pred_prob,axis=1)
    return {
            "response": res[str(pred_label.item())]
        }



# result is the playlist name (playlist id)

