
# Spam Text Classification

This project is focused on classifying text messages as either 'spam' or 'ham' (not spam) using machine learning techniques. The dataset used consists of text messages labeled as spam or ham. The model is built using TensorFlow and Keras, and it leverages an LSTM (Long Short-Term Memory) neural network to perform the classification.

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

## Usage

1. **Data Preprocessing:**
   Run the notebook cells to load and preprocess the data.

2. **Model Training:**
   Train the LSTM model by running the notebook cells. The training process includes compiling the model, fitting it to the training data, and evaluating it on the validation data.

3. **Hyperparameter Tuning:**
   Perform hyperparameter tuning to optimize the model parameters.

4. **Model Evaluation:**
   Evaluate the model's performance using confusion matrix and ROC curve.

5. **Model Saving:**
   Save the trained model using the `model.save()` method.

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
