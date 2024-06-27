import pandas as pd
import nltk
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('isri')
from nltk.stem.isri import ISRIStemmer
nltk.download('punkt')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


data = pd.read_csv('../data/preprocessed_data_clean_after_augmentation.csv')
textBody = data.get('text')
label = data.get('label')
print(data.head())
# Tokenization
data['tokens'] = data['text'].apply(lambda x: nltk.word_tokenize(x))

# Feature Extraction with TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in data['tokens']])

# # Feature Extraction with N-grams (using CountVectorizer)
# ngram_vectorizer = CountVectorizer(ngram_range=(2, 3))
# ngram_matrix = ngram_vectorizer.fit_transform([' '.join(tokens) for tokens in data['tokens']])
#
# # # Feature Extraction with Bag of Words (BoW)
# bow_vectorizer = CountVectorizer()
# bow_matrix = bow_vectorizer.fit_transform([' '.join(tokens) for tokens in data['tokens']])
#
# # Feature Extraction with Word2Vec
# # Assuming tokens are already preprocessed and tokenized
# sentences = data['tokens'].tolist()
# word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Create a function to convert tokens to Word2Vec embeddings
# def get_word2vec_embeddings(tokens):
#     embeddings = []
#     for token in tokens:
#         if token in word2vec_model.wv:
#             embeddings.append(word2vec_model.wv[token])
#     return np.mean(embeddings, axis=0) if embeddings else np.zeros(word2vec_model.vector_size)
#
# # Apply the Word2Vec feature extraction to each row in the DataFrame
# word2vec_embeddings = np.array([get_word2vec_embeddings(tokens) for tokens in data['tokens']])
#
# scaler = MinMaxScaler()
# word2vec_embeddings_normalized = scaler.fit_transform(word2vec_embeddings)

# Concatenate the feature matrices
# final_feature_matrix = np.concatenate([bow_matrix.toarray(), word2vec_embeddings_normalized], axis=1)

# Now, 'final_feature_matrix' contains the combined features from TF-IDF, N-grams, and Word2Vec for each document.nts.

#Using TF-IDF
# Assuming 'final_feature_matrix' is your feature matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data['label'], test_size=0.2, random_state=42)

# # Naive Bayes Classifier
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X_train, y_train)
# nb_predictions = nb_classifier.predict(X_test)
#
# print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
# print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(C=2, class_weight='balanced', kernel='rbf')
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

svm_model_file='../models/svm_model.joblib'
dump(svm_classifier, svm_model_file)
# # Support Vector Machine (LSVM) Classifier
# lsvm_classifier = LinearSVC(class_weight='balanced')
# lsvm_classifier.fit(X_train, y_train)
# lsvm_predictions = lsvm_classifier.predict(X_test)
#
# print("LSVM Accuracy:", accuracy_score(y_test, lsvm_predictions))
# print("LSVM Classification Report:\n", classification_report(y_test, lsvm_predictions))

# Random Forest Classifier
# rf_classifier = RandomForestClassifier(class_weight='balanced')
# rf_classifier.fit(X_train, y_train)
# rf_predictions = rf_classifier.predict(X_test)
#
# print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
# print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

#USING N-GRAM
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(ngram_matrix, data['label'], test_size=0.20, random_state=42)
#
# # Naive Bayes Classifier
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X_train, y_train)
# nb_predictions = nb_classifier.predict(X_test)
#
# print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
# print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))
#
# # Support Vector Machine (SVM) Classifier
# svm_classifier = SVC(C=2, class_weight='balanced', kernel='rbf')
# svm_classifier.fit(X_train, y_train)
# svm_predictions = svm_classifier.predict(X_test)
#
# print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
# print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
#
# # Support Vector Machine (LSVM) Classifier
# lsvm_classifier = LinearSVC(class_weight='balanced')
# lsvm_classifier.fit(X_train, y_train)
# lsvm_predictions = lsvm_classifier.predict(X_test)
#
# print("LSVM Accuracy:", accuracy_score(y_test, lsvm_predictions))
# print("LSVM Classification Report:\n", classification_report(y_test, lsvm_predictions))
#
# # Random Forest Classifier
# rf_classifier = RandomForestClassifier(class_weight='balanced')
# rf_classifier.fit(X_train, y_train)
# rf_predictions = rf_classifier.predict(X_test)
#
# print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
# print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

# Using Word2Vec
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(word2vec_embeddings_normalized, data['label'], test_size=0.2, random_state=42)
#
# # Naive Bayes Classifier
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X_train, y_train)
# nb_predictions = nb_classifier.predict(X_test)
#
# print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
# print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions, zero_division=1))
#
# # Support Vector Machine (SVM) Classifier
# svm_classifier = SVC()
# svm_classifier.fit(X_train, y_train)
# svm_predictions = svm_classifier.predict(X_test)
#
# print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
# print("SVM Classification Report:\n", classification_report(y_test, svm_predictions, zero_division=1))
#
# # Support Vector Machine (LSVM) Classifier
# lsvm_classifier = LinearSVC(class_weight='balanced')
# lsvm_classifier.fit(X_train, y_train)
# lsvm_predictions = lsvm_classifier.predict(X_test)
#
# print("LSVM Accuracy:", accuracy_score(y_test, lsvm_predictions))
# print("LSVM Classification Report:\n", classification_report(y_test, lsvm_predictions))
#
# # Random Forest Classifier
# rf_classifier = RandomForestClassifier()
# rf_classifier.fit(X_train, y_train)
# rf_predictions = rf_classifier.predict(X_test)
#
# print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
# print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions, zero_division=1))

# Using BOW
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(bow_matrix, data['label'], test_size=0.30, random_state=42)
#
# # Naive Bayes Classifier
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X_train, y_train)
# nb_predictions = nb_classifier.predict(X_test)
#
# print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
# print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))
#
# # Support Vector Machine (SVM) Classifier
# svm_classifier = SVC(C=2, class_weight='balanced', kernel='rbf')
# svm_classifier.fit(X_train, y_train)
# svm_predictions = svm_classifier.predict(X_test)
#
# print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
# print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
#
# # Support Vector Machine (LSVM) Classifier
# lsvm_classifier = LinearSVC(class_weight='balanced')
# lsvm_classifier.fit(X_train, y_train)
# lsvm_predictions = lsvm_classifier.predict(X_test)
#
# print("LSVM Accuracy:", accuracy_score(y_test, lsvm_predictions))
# print("LSVM Classification Report:\n", classification_report(y_test, lsvm_predictions))
#
# # Random Forest Classifier
# rf_classifier = RandomForestClassifier(class_weight='balanced')
# rf_classifier.fit(X_train, y_train)
# rf_predictions = rf_classifier.predict(X_test)
#
# print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
# print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))