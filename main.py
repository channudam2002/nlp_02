from fastapi import FastAPI
from pydantic import BaseModel
from model import predict
from proprocess import preprocessing_text, feature_engineering

app = FastAPI()

class TextAnalytic(BaseModel):
    text: str

@app.post("/predict/")
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

@app.post("/analyze")
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