"""
inference.py - Test the model from Hugging Face
"""

from transformers import pipeline

def load_model():
    """Load the model from Hugging Face"""
    model_name = "bhattaraisuman/nepali-sentiment-mbert"
    
    print("🚀 Loading Nepali Sentiment Model from Hugging Face...")
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name
    )
    print("✅ Model loaded successfully!")
    return classifier

def predict_sentiment(classifier, text):
    """Predict sentiment for given text"""
    result = classifier(text)[0]
    
    # Map labels to readable names
    label_map = {
        'LABEL_0': 'Negative 😠',
        'LABEL_1': 'Neutral 😐', 
        'LABEL_2': 'Positive 😊'
    }
    
    sentiment = label_map.get(result['label'], result['label'])
    confidence = result['score']
    
    return sentiment, confidence

def main():
    classifier = load_model()
    
    print("\n💬 Nepali Sentiment Analysis")
    print("   Model: bhattaraisuman/nepali-sentiment-mbert")
    print("   (Type 'quit' to exit)\n")
    
    while True:
        text = input("📝 Enter Nepali text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
            
        if text:
            try:
                sentiment, confidence = predict_sentiment(classifier, text)
                print(f"🎯 Sentiment: {sentiment}")
                print(f"📊 Confidence: {confidence:.2%}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()