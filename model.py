import tensorflow as tf
import warnings
import pickle
from keras.layers.experimental.preprocessing import TextVectorization
from proprocess import preprocessing_text, feature_engineering

warnings.filterwarnings("ignore", category=FutureWarning)

LOEADED_MODEL = tf.keras.models.load_model('models/sentimental.h5')
print(LOEADED_MODEL.summary())
print(f"Model inputs: {LOEADED_MODEL.inputs}")
print(f"Model outputs: {LOEADED_MODEL.outputs}")


def text_vectorize(text):
    cleaned_text = preprocessing_text(text)
    from_file = pickle.load(open("vectorizers/vectorizer.pkl", "rb"))
    vectorizer = TextVectorization.from_config(from_file['config'])
    vectorizer.set_weights(from_file['weights'])
    text = vectorizer([cleaned_text])
    return text

def predict(text):
    X_text =  text_vectorize(text)
    X_numerical = feature_engineering(text)
    result = LOEADED_MODEL.predict([X_text, X_numerical])
    return result
