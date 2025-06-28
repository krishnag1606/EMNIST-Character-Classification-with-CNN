## EMNIST Character Classification with CNN

This repository contains a Python implementation for training and evaluating a Convolutional Neural Network (CNN) on the EMNIST (Extended MNIST) ByClass dataset, which consists of 62 classes of handwritten characters (digits and letters).

##Co-Authors

Krishna Gupta                                                                      https://github.com/krishnag1606
Shashwat Kumar Banal                                                               https://github.com/shashwatb23

### Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [Dependencies](#dependencies)
* [License](#license)

---

## Project Overview

This project demonstrates how to:

1. Load and preprocess the EMNIST ByClass dataset.
2. Build a simple CNN using TensorFlow and Keras.
3. Train the model on the dataset.
4. Evaluate model performance using common classification metrics.

## Dataset

* **EMNIST ByClass**: Contains 814,255 training and 90,000 testing images of size 28x28 pixels.
* **Classes**: 62 classes (10 digits + 52 uppercase and lowercase letters).

Dataset source: [Kaggle EMNIST Dataset](https://www.kaggle.com/crawford/emnist)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/emnist-cnn.git
   cd emnist-cnn
   ```
2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
   ```
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the EMNIST ByClass CSV files from Kaggle and place them in a folder named `input/emnist`.
2. Run the training script:

   ```bash
   python train_emnist_cnn.py
   ```
3. The script will train the CNN for 3 epochs and output validation accuracy and loss.

## Model Architecture

| Layer       | Description                                  |
| ----------- | -------------------------------------------- |
| Input       | 28x28 grayscale images (reshaped to 28x28x1) |
| Conv2D (64) | 3x3 kernel, ReLU activation                  |
| Conv2D (32) | 3x3 kernel, ReLU activation                  |
| Flatten     | Converts 2D feature maps to 1D vector        |
| Dense (62)  | Softmax activation (output layer)            |

## Training

* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy
* **Epochs**: 3
* **Batch Size**: Default (as per Keras, typically 32)

## Evaluation Metrics

After training, the following metrics can be computed on test data:

* **Accuracy**: Overall correctness of predictions.
* **Precision**: Ratio of true positives to all predicted positives.
* **Recall**: Ratio of true positives to all actual positives.
* **F1 Score**: Harmonic mean of precision and recall.

Example code snippet:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("Precision:", precision_score(y_test, y_pred_classes, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_classes, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_classes, average='weighted'))
```

## Results

| Metric    | Value (example) |
| --------- | --------------- |
| Accuracy  | 85%             |
| Precision | 0.86            |
| Recall    | 0.85            |
| F1 Score  | 0.85            |

*Note: These values will vary depending on training runs.*

## Dependencies

* Python 3.7+
* pandas
* numpy
* matplotlib
* tensorflow
* keras
* scikit-learn

A full list is provided in `requirements.txt`.
