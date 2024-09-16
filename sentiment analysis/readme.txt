
---

# Sentiment Analysis with Ensemble of Pretrained Models

This notebook demonstrates an ensemble sentiment analysis approach by leveraging the predictions from multiple pre-trained models (BERT, RoBERTa, and DistilBERT). The ensemble approach uses a majority voting mechanism to determine the final sentiment label for each social media post.

## Features

- **Ensemble Sentiment Analysis**: Combines predictions from three pre-trained language models: BERT, RoBERTa, and DistilBERT.
- **Majority Voting**: Determines the final sentiment based on the majority of predicted labels from the three models.
- **Pretrained Models**: Uses Hugging Face's transformers library to load and run the models.

## How It Works

1. **Input Data**: A list of social media posts is used as input for sentiment analysis.
   
2. **Model Predictions**:
   - The posts are passed through three models: BERT, RoBERTa, and DistilBERT.
   - Each model returns a sentiment label for each post (e.g., "positive", "negative", "neutral").

3. **Ensemble Approach**:
   - The sentiment labels from each model are combined using a majority voting system. The final sentiment label for each post is the one that appears most often across the three models.

4. **Results**:
   - The ensemble sentiment label, along with individual model predictions, is displayed for each post.

## Usage

### Data

The social media posts are provided in a Python list called `social_media_posts`. You can modify this list with your own data to analyze different posts.

Example:

```python
social_media_posts = [
    "I love this product! It works really well and has great customer support.",
    "Worst experience ever! I will never shop here again.",
    "The service was okay, but the delivery was late.",
    "I'm so happy with my purchase! Totally recommend it.",
    "This was a terrible decision, very disappointed with the quality."
]
```

### Function

The main function that performs the ensemble sentiment analysis is `ensemble_sentiment_analysis(posts)`. It takes a list of posts as input and returns the combined sentiment results.

```python
def ensemble_sentiment_analysis(posts):
    combined_results = []
    for post in posts:
        # Get predictions from all models
        bert_result = bert_classifier(post)[0]
        roberta_result = roberta_classifier(post)[0]
        distilbert_result = distilbert_classifier(post)[0]

        # Combine results using majority voting
        sentiments = [bert_result['label'], roberta_result['label'], distilbert_result['label']]
        majority_sentiment = max(set(sentiments), key=sentiments.count)

        combined_results.append({
            "Post": post,
            "BERT Sentiment": bert_result,
            "RoBERTa Sentiment": roberta_result,
            "DistilBERT Sentiment": distilbert_result,
            "Ensembled Sentiment": majority_sentiment
        })
    
    return combined_results
```

### Running the Analysis

Once the function is defined, you can run it using the example social media posts:

```python
results = ensemble_sentiment_analysis(social_media_posts)
for result in results:
    print(f"Post: {result['Post']}")
    print(f"BERT: {result['BERT Sentiment']}")
    print(f"RoBERTa: {result['RoBERTa Sentiment']}")
    print(f"DistilBERT: {result['DistilBERT Sentiment']}")
    print(f"Ensembled Sentiment: {result['Ensembled Sentiment']}\n")
```

### Output

The output for each post will display:
- The original post.
- The sentiment predicted by BERT, RoBERTa, and DistilBERT.
- The ensembled sentiment based on majority voting.

## Requirements

To run this notebook, ensure you have the following dependencies installed:

- `transformers`
- `torch`
- `numpy`
- `pandas`

You can install the required libraries using pip:

```bash
pip install transformers torch numpy pandas
```

## Conclusion

This notebook showcases an effective way of combining multiple sentiment analysis models to produce a more reliable sentiment prediction. By leveraging the strengths of different models and using a majority voting mechanism, the overall prediction accuracy can be improved.

---
