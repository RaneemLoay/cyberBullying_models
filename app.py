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
# from notebooks.data_preprocessing import remove_emoji,arabic_diacritics,stop_words,punctuations_list
from notebooks.data_preprocessing import preprocess


app = Flask(__name__)

# Load tokenizer
with open('static/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load models
model_lstm = load_model('models/lstm_model.h5')
model_bilstm = load_model('models/bilstm_model.h5')
model_gru = load_model('models/gru_model.h5')
model_cnn = load_model('models/cnn_model.h5')

MAX_SEQUENCE_LENGTH = 100
# # Define preprocessing functions
# def preprocess(text):
#     # Define preprocessing steps similar to your notebook
#     data_preprocessing.preprocess(text)
#     # Example: Remove numbers and special characters
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#
#     # Tokenize text
#     tokens = word_tokenize(text)
#
#     # Remove stop words (example)
#     stop_words = set(stopwords.words('english'))
#     tokens = [w for w in tokens if not w in stop_words]
#
#     # Tokenize and pad sequences
#     sequences = tokenizer.texts_to_sequences([tokens])
#     padded_sequences = pad_sequences(sequences, maxlen=100)
#
#     return padded_sequences
# def preprocess(text):
#     if pd.isnull(text):
#         return ''
#     # Remove punctuations
#     text = text.replace('_', ' ')
#     text = "".join([char for char in text if char not in punctuations_list])
#
#     # Remove emojis
#     text = remove_emoji(text)
#
#     # Remove numbers
#     text = re.sub(r"[0123456789٠١٢٣٤٥٦٧٨٩]", '', text)
#
#     # Remove English letters
#     text = re.sub(r"[a-zA-Z]", '', text)
#
#     # Remove diacritics
#     text = re.sub(arabic_diacritics, '', text)
#
#     # Normalize text
#     text = re.sub(r"[ٱإأآا]", "ا", text)
#     text = re.sub(r"ى", "ي", text)
#     text = re.sub(r"ؤ", "ء", text)
#     text = re.sub(r"ئ", "ء", text)
#     text = re.sub(r"ة", "ه", text)
#     text = re.sub(r"گ", "ك", text)
#     text = re.sub(r"ک", "ك", text)
#     text = re.sub(r"؏", "ع", text)
#
#     # Remove elongation
#     text = re.sub(r'(.)\1+', r"\1\1", text)
#
#     # Tokenize
#     tokens = word_tokenize(text)
#
#     # Remove stop words
#     text = ' '.join([word for word in tokens if word not in stop_words])
#     if pd.isnull(text):
#         return ''
#     return text


# Define routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

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

        predicted_class = "bullying" if pred_cnn[0][0] < 0.5 else "not bullying"
        return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    # app.run(debug=True)