A comprehensive sentiment analysis system for Nepali text, fine-tuned on BERT multilingual model. Classifies Nepali sentences into Negative, Neutral, or Positive sentiments with medium (~64%) accuracy.
**Features**
Fine-tuned BERT Model: Optimized bert-base-multilingual-cased for Nepali language
Multiple Interfaces: Command-line, Web App, and REST API
Medium Accuracy: ~64% accuracy on test data
Easy Integration: Ready-to-use via Hugging Face Hub
Production Ready: FastAPI backend and Streamlit frontend

**Prerequisites**
python >= 3.8 (best for 3.11.0)
pip install -r requirements.txt

**Installation**
# Clone the repository
git clone https://github.com/sumanbhattarai58/Nepali_sentiment_analysis.git
cd Nepali_sentiment_analysis

**Usage**
**Option 1: Command Line Interface**
           python inference.py
Then enter Nepali text like:
"यो धेरै राम्रो छ!" → Positive 😊
"यो सामान्य छ" → Neutral 😐
"यो नराम्रो छ" → Negative 😠

**Option 2: REST API (FastAPI)**
          uvicorn app:app --reload --port 8000
          Visit http://localhost:8000/docs for interactive API documentation.

**Option 3: Web Interface (Streamlit)**
           streamlit run streamlit_app.py
           Visit http://localhost:8501 for interactive web app.

**Model Details**
**Aspect**	                                **Details**
Base Model	                            bert-base-multilingual-cased
Task	                                  Text Classification / Sentiment Analysis
Language	                              Nepali
Labels	                                Negative (0), Neutral (1), Positive (2)
Training Data	                          Shushant/NepaliSentiment (Hugging Face)
Accuracy	                              ~64%

**Training**
The model was fine-tuned with the following hyperparameters:
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
OPTIMIZER = AdamW
SCHEDULER = Cosine

**🔗 Links**
Hugging Face Model: bhattaraisuman/nepali-sentiment-mbert
Dataset: Shushant/NepaliSentiment
