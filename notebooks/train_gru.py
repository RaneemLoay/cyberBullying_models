import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D ,GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
# Load preprocessed data
df_clean = pd.read_csv('../data/preprocessed_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_clean['text'], df_clean['label'], test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
max_len = 100  # or your chosen sequence length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Convert labels to categorical format if needed
y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)

from tensorflow.keras.layers import GRU

# Define the GRU model
model_gru = Sequential()
model_gru.add(Embedding(input_dim=5000, output_dim=100, input_length=max_len))
model_gru.add(SpatialDropout1D(0.3))
model_gru.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))
model_gru.add(Dense(2, activation='softmax'))

# Compile the model
model_gru.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history_gru = model_gru.fit(X_train_pad, y_train_enc, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss_gru, accuracy_gru = model_gru.evaluate(X_test_pad, y_test_enc, verbose=2)
print(f'GRU Accuracy: {accuracy_gru * 100:.2f}%')

# Save the model
model_gru.save('../models/gru_model.h5')
