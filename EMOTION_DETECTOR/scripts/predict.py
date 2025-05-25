import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

model_path = os.path.join(base_dir, 'models', 'emotion_detector_model.h5')
model = tf.keras.models.load_model(model_path)
print("Model loaded.")

train_csv_path = os.path.join(base_dir, 'dataset', 'train.csv')
train_df = pd.read_csv(train_csv_path)
texts = train_df["text"].tolist()
labels = train_df["label"].tolist()

tokenizer_path = os.path.join(base_dir, "models", "tokenizer.pkl")
try:
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded.")
except FileNotFoundError:
    print("Tokenizer not found, fitting new tokenizer...")
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle)
    print("Tokenizer created and saved.")

label_encoder = LabelEncoder()
label_encoder.fit(labels)
print("Label classes:", label_encoder.classes_)

def preprocess_and_predict(text, tokenizer=tokenizer, model=model, max_len=50):
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding="post")
        prediction = model.predict(padded_seq)
        pred_class_idx = np.argmax(prediction)
        emotion = label_encoder.inverse_transform([pred_class_idx])[0]
        return emotion
    except Exception as e:
        return f"Prediction error: {str(e)}"