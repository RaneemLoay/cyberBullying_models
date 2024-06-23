import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# Load preprocessed data
df_clean = pd.read_csv('../data/preprocessed_data.csv')

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
max_len = 100  # you can choose an appropriate length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Convert labels to numerical format
y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)

# Print shapes for verification
print(f'Training data shape: {X_train_pad.shape}')
print(f'Training labels shape: {y_train_enc.shape}')
print(f'Test data shape: {X_test_pad.shape}')
print(f'Test labels shape: {y_test_enc.shape}')

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data['label']), y=data['label'])
class_weights = {i : class_weights[i] for i in range(len(class_weights))}

from tensorflow.keras.layers import Bidirectional

# Build the model
embedding_dim = 300
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_bilstm = model.fit(X_train_pad, y_train_enc, epochs=10, batch_size=32,validation_split=0.1, callbacks=[early_stopping], class_weight=class_weights)

# Evaluate the model
accuracy = model.evaluate(X_test_pad, y_test_enc, verbose=2)[1]
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Generate classification report
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes, target_names=['not bullying', 'bullying']))

# Save the model
model_bilstm.save('../models/bilstm_model.h5')
