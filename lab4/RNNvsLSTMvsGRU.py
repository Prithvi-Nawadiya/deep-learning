
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding
from sklearn.metrics import accuracy_score


# Load top 10,000 words
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


max_len = 200

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


def build_model(model_type):
    model = Sequential()
    
    # Embedding Layer
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len))
    
    if model_type == "RNN":
        model.add(SimpleRNN(32))
    elif model_type == "LSTM":
        model.add(LSTM(32))
    elif model_type == "GRU":
        model.add(GRU(32))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


results = {}

for model_type in ["RNN", "LSTM", "GRU"]:
    print(f"\nTraining {model_type}...")
    
    model = build_model(model_type)
    
    start_time = time.time()
    model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    acc = accuracy_score(y_test, y_pred)
    
    results[model_type] = {
        "Accuracy": acc,
        "Time": training_time
    }
    
    print(f"{model_type} -> Accuracy: {acc:.4f}, Time: {training_time:.2f}s")


models = list(results.keys())
acc_values = [results[m]["Accuracy"] for m in models]
time_values = [results[m]["Time"] for m in models]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.bar(models, acc_values)
plt.title("Accuracy Comparison")

plt.subplot(1,2,2)
plt.bar(models, time_values)
plt.title("Training Time")

plt.show()
