# Spam Text Classification

This project is focused on classifying text messages as either 'spam' or 'ham' (not spam) using machine learning techniques. The dataset used consists of text messages labeled as spam or ham. The model is built using TensorFlow and Keras, and it leverages an LSTM (Long Short-Term Memory) neural network to perform the classification.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Hotzones](#hotzones)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to classify text messages as 'spam' or 'ham' using an LSTM neural network. The dataset used is composed of labeled text messages, and the project covers the entire workflow from data loading and preprocessing to model training, hyperparameter tuning, evaluation, and saving the trained model for future use.

## Project Structure

- **Data Loading and Preprocessing:**
  - Load the dataset.
  - Clean and preprocess the text data, including tokenization and padding sequences.

- **Model Building:**
  - Define an LSTM model architecture.
  - Compile the model with appropriate loss function and optimizer.

- **Model Training:**
  - Train the model on the preprocessed text data.
  - Evaluate the model on validation data.

- **Hyperparameter Tuning:**
  - Perform hyperparameter tuning to find the best parameters for the model.

- **Model Evaluation:**
  - Generate confusion matrix and ROC curve to evaluate the performance of the model.

- **Model Saving:**
  - Save the trained model for future use.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

### Setting Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Here's how you can set it up:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing:**
   - Run the notebook cells to load and preprocess the data.

2. **Model Training:**
   - Train the LSTM model by running the notebook cells. The training process includes compiling the model, fitting it to the training data, and evaluating it on the validation data.

3. **Hyperparameter Tuning:**
   - Perform hyperparameter tuning to optimize the model parameters.

4. **Model Evaluation:**
   - Evaluate the model's performance using confusion matrix and ROC curve.

5. **Model Saving:**
   - Save the trained model using the `model.save()` method.

### Example Code Snippets

#### Data Preprocessing

```python
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
data = pd.read_csv('spam_ham_dataset.csv')

# Clean and preprocess text data
texts = data['text'].values
labels = data['label'].values

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)
```

#### Model Building and Training

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Define LSTM model architecture
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=5, batch_size=64, validation_split=0.2)
```

#### Model Evaluation

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Predict labels
predictions = model.predict(padded_sequences)
predictions = (predictions > 0.5).astype(int)

# Confusion matrix
conf_matrix = confusion_matrix(labels, predictions)

# ROC curve
fpr, tpr, _ = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

#### Model Saving

```python
# Save the trained model
model.save('spam_classifier_model.h5')
```

## Evaluation Metrics

- **Confusion Matrix:**
  Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

- **ROC Curve:**
  Illustrates the trade-off between true positive rate and false positive rate across different thresholds.

## Hotzones

The following sections are considered hotzones of this project, where most significant operations and analyses occur:

1. **Data Preprocessing:**
   Key steps include tokenizing and padding sequences to prepare the data for model training.

2. **Model Building and Training:**
   The core of the project, where the LSTM model is defined, compiled, and trained on the data.

3. **Hyperparameter Tuning:**
   Critical for optimizing the model's performance by finding the best set of hyperparameters.

4. **Model Evaluation:**
   Generating confusion matrix and ROC curve to evaluate the model's performance.

5. **Model Saving:**
   Saving the trained model for future predictions and use.

## Results

The trained LSTM model achieves a high accuracy in classifying text messages as spam or ham. The confusion matrix and ROC curve provide detailed insights into the model's performance.

## Conclusion

This project demonstrates a complete workflow for building, training, and evaluating a text classification model using LSTM neural networks. It provides a solid foundation for further improvements and real-world applications.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
