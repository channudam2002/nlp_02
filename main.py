import os
import sys
from fastapi import FastAPI
from fastapi.logger import logger
import logging
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from model import predict
from proprocess import preprocessing_text, feature_engineering
from typing import ClassVar


class TextAnalytic(BaseModel):
    text: str

class Settings(BaseSettings):
    BASE_URL: ClassVar[str]= "http://127.0.0.1:8000"
    USE_NGROK: ClassVar[bool] = os.environ.get("USE_NGROK", "False") == "True"


settings = Settings()
app = FastAPI()
logger.setLevel(logging.INFO)


def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass

if settings.USE_NGROK and os.environ.get("NGROK_AUTHTOKEN"):
    from pyngrok import ngrok
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8000"
    public_url = ngrok.connect(port).public_url
    print(public_url)
    logger.info(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    Settings.BASE_URL = public_url
    init_webhooks(public_url)

@app.post("/predict/", tags=["sentimental prediction"])
def prediction(request: TextAnalytic):
    result = predict(request.text)
    return {
        'data':{
            'result':{
                'positive_review_rate': round(float(result[0]), 2),
                'negative_review_rate': round(float(1-result[0]), 2)
            }
        }
    } 

@app.post("/analyze", tags=["sentimental analysis"])
def analyze(request: TextAnalytic):
    features = feature_engineering(request.text)
    return {
        'data':{
            'result':{
                'raw_text': request.text,
                'cleaned_text': preprocessing_text(request.text),
                'features': {
                    'positive_word_count': float(features[0][0]),
                    'negative_word_count': float(features[0][1]),
                    'pronoun_count': float(features[0][3]),
                    'length': float(features[0][5]),
                    "is_contained_exclamation": bool(features[0][4]),
                    "is_contained_no": bool(features[0][2])
                }
            }
        }
    }