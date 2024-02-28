from fastapi import FastAPI
from pydantic import BaseModel
from model import predict
from proprocess import preprocessing_text

app = FastAPI()

class TextAnalytic(BaseModel):
    text: str

@app.post("/preprocessing")
def preprocessing(request: TextAnalytic):
    return {
        'data':{
            'result':{
                'text': preprocessing_text(request.text)
            }
        }
    }

@app.post("/analyze/")
def analyze(request: TextAnalytic):
    return predict(request.text) 