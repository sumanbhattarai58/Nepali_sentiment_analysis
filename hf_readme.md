---
language:
- ne
tags:
- sentiment-analysis
- nepali
- bert
- multilingual
license: apache-2.0
datasets:
- Shushant/NepaliSentiment
---

# Nepali Sentiment Analysis with mBERT

This model is fine-tuned `bert-base-multilingual-cased` for sentiment analysis on Nepali text. It classifies sentences into **Negative (0)**, **Neutral (1)**, or **Positive (2)** sentiments.

## Model Details
- **Base Model:** bert-base-multilingual-cased
- **Task:** Text Classification / Sentiment Analysis
- **Language:** Nepali
- **Number of Labels:** 3
- **Training Data:** Shushant/NepaliSentiment dataset
- **License:** Apache 2.0

## Usage

### Using Pipeline (Simple)
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification", 
    model="bhattaraisuman/nepali-sentiment-mbert"
)

results = classifier("यो धेरै राम्रो छ!")
print(results)
# [{'label': 'LABEL_2', 'score': 0.95}]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bhattaraisuman/nepali-sentiment-mbert")
model = AutoModelForSequenceClassification.from_pretrained("bhattaraisuman/nepali-sentiment-mbert")

# Input text
text = "तपाईंको परियोजना राम्रो छ"

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get results
predicted_class = predictions.argmax().item()
confidence = predictions.max().item()

# Map to sentiment labels
sentiments = ['Negative', 'Neutral', 'Positive']
print(f"Text: {text}")
print(f"Sentiment: {sentiments[predicted_class]}")
print(f"Confidence: {confidence:.2%}")

# Optional: See probabilities for all classes
print("\nAll probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"  {sentiments[i]}: {prob:.2%}")