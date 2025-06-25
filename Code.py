import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Constants
TRAIN_CSV = '/kaggle/input/emnist/emnist-byclass-train.csv'
TEST_CSV = '/kaggle/input/emnist/emnist-byclass-test.csv'
IMAGE_SHAPE = (28, 28, 1)
NUM_CLASSES = 62
EPOCHS = 3
BATCH_SIZE = 32

# Load data
train_df = pd.read_csv(TRAIN_CSV, header=None)
test_df = pd.read_csv(TEST_CSV, header=None)

# Preprocess features
x_train = train_df.iloc[:, 1:].values.astype('float32') / 255.0
x_test = test_df.iloc[:, 1:].values.astype('float32') / 255.0
x_train = x_train.reshape(-1, *IMAGE_SHAPE)
x_test = x_test.reshape(-1, *IMAGE_SHAPE)

# Preprocess labels
y_train = train_df.iloc[:, 0].values
y_test = test_df.iloc[:, 0].values

y_train_enc = tf.one_hot(y_train, depth=NUM_CLASSES)
y_test_enc = tf.one_hot(y_test, depth=NUM_CLASSES)

# Build model
model = Sequential([
    Conv2D(64, kernel_size=3, activation='relu', input_shape=IMAGE_SHAPE),
    Conv2D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    x_train, y_train_enc,
    validation_data=(x_test, y_test_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Save model
model.save('emnist_cnn_model.h5')
print("Model saved to emnist_cnn_model.h5")
