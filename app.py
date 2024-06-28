import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model
from joblib import load
from notebooks.data_preprocessing import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load tokenizer and models globally to avoid reloading on every request
tokenizer = None
model_lstm = None
model_bilstm = None
model_gru = None
model_cnn = None
model_svm = None

def load_resources():
    global tokenizer, tfidf_vectorizer, model_lstm, model_bilstm, model_gru, model_cnn ,model_svm
    if tokenizer is None:
        with open('./static/tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer()
    if model_lstm is None:
        model_lstm = load_model('./models/lstm_model.h5')
    if model_bilstm is None:
        model_bilstm = load_model('./models/bilstm_model.h5')
    if model_gru is None:
        model_gru = load_model('./models/gru_model.h5')
    if model_cnn is None:
        model_cnn = load_model('./models/cnn_model.h5')
    if model_svm is None:
        model_svm = load('./models/svm_model.joblib')

load_resources()

MAX_SEQUENCE_LENGTH = 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # text = request.form['text']
        data = request.get_json()
        text = data['text']
        try:
            # Preprocess text
            processed_text = preprocess(text)
            print(processed_text)

            # Tokenize and pad sequence
            sequence = tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
            padded_sequence = np.array(padded_sequence)

            # Make predictions using loaded models
            pred_lstm = model_lstm.predict(padded_sequence)
            pred_bilstm = model_bilstm.predict(padded_sequence)
            pred_gru = model_gru.predict(padded_sequence)
            pred_cnn = model_cnn.predict(padded_sequence)
            
            new_text_tfidf = tfidf_vectorizer.transform(processed_text)
            pred_svm = model_svm.predict(new_text_tfidf)

            predicted_class = "bullying" if pred_svm[0][0] < 0.5 else "not bullying"
            print(f"text: {text}")
            print(f"prediction: {predicted_class}")
            return jsonify({'prediction': predicted_class})
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Error during prediction'}), 505

if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 8080))
    # app.run(debug=True, host='0.0.0.0', port=port)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
