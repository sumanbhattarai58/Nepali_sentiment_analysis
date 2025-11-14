"""
FastAPI Inference Server for Nepali Sentiment Analysis
======================================================
This API serves the trained BERT model for real-time sentiment prediction.

Endpoints:
- POST /predict - Single text prediction
- POST /predict_batch - Batch predictions
- GET /health - Health check
- GET / - API documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import time
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
class Config:
    MODEL_PATH = "bhattaraisuman/Nepali_sentiment_bert"  # Path to saved model
    MAX_LENGTH = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    
config = Config()

# ============================================
# PYDANTIC MODELS (Request/Response)
# ============================================
class TextInput(BaseModel):
    text: str = Field(..., description="Nepali text for sentiment analysis", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "यो फिल्म एकदम राम्रो छ 😊"
            }
        }

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of Nepali texts", min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "यो फिल्म एकदम राम्रो छ 😊",
                    "यो खराब सेवा हो",
                    "ठीक छ, केही खास छैन"
                ]
            }
        }

class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict
    processing_time: float

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_processing_time: float
    count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

# ============================================
# MODEL MANAGER
# ============================================
class SentimentModel:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.label_names = ['Negative', 'Neutral', 'Positive']
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single(self, text: str) -> dict:
        """Predict sentiment for a single text"""
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        # Prepare response
        processing_time = time.time() - start_time
        
        return {
            "text": text,
            "sentiment": self.label_names[pred_idx],
            "confidence": round(confidence, 4),
            "probabilities": {
                "Negative": round(probs[0].item(), 4),
                "Neutral": round(probs[1].item(), 4),
                "Positive": round(probs[2].item(), 4)
            },
            "processing_time": round(processing_time, 4)
        }
    
    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Predict sentiment for multiple texts"""
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), config.BATCH_SIZE):
            batch_texts = texts[i:i + config.BATCH_SIZE]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_indices = torch.argmax(probs, dim=1)
            
            # Process results
            for j, text in enumerate(batch_texts):
                pred_idx = pred_indices[j].item()
                confidence = probs[j][pred_idx].item()
                
                results.append({
                    "text": text,
                    "sentiment": self.label_names[pred_idx],
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "Negative": round(probs[j][0].item(), 4),
                        "Neutral": round(probs[j][1].item(), 4),
                        "Positive": round(probs[j][2].item(), 4)
                    }
                })
        
        return results

# ============================================
# FASTAPI APPLICATION
# ============================================
app = FastAPI(
    title="Nepali Sentiment Analysis API",
    description="Real-time sentiment analysis for Nepali text using BERT",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model (will be loaded on startup)
sentiment_model: Optional[SentimentModel] = None

# ============================================
# STARTUP & SHUTDOWN EVENTS
# ============================================
@app.on_event("startup")
async def startup_event():
    """Start model loading in background"""
    global sentiment_model
    print("="*60)
    print("🚀 Starting Nepali Sentiment Analysis API")
    print("="*60)
    print("⏳ Model will load in background...")
    print("="*60)
    
    # Load model in background (non-blocking)
    import asyncio
    asyncio.create_task(load_model_background())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n👋 Shutting down API...")

# ADD THIS NEW FUNCTION HERE:
async def load_model_background():
    """Load model in background"""
    global sentiment_model
    try:
        sentiment_model = SentimentModel(
            model_path=config.MODEL_PATH,
            device=config.DEVICE
        )
        print("✅ Model loaded and ready!")
        print("="*60)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy" if sentiment_model is not None else "loading",
        "model_loaded": sentiment_model is not None,
        "device": config.DEVICE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for a single Nepali text.
    
    Returns sentiment (Negative/Neutral/Positive) with confidence scores.
    """
    if sentiment_model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please wait...")
    
    try:
        result = sentiment_model.predict_single(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple Nepali texts.
    
    Efficiently processes up to 100 texts at once.
    """
    if sentiment_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(input_data.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts allowed per batch")
    
    try:
        start_time = time.time()
        results = sentiment_model.predict_batch(input_data.texts)
        
        # Add processing time to each result
        for result in results:
            result["processing_time"] = 0.0  # Individual times not tracked in batch
        
        total_time = time.time() - start_time
        
        return {
            "predictions": results,
            "total_processing_time": round(total_time, 4),
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/info")
async def get_info():
    """Get API and model information"""
    return {
        "api_name": "Nepali Sentiment Analysis API",
        "version": "1.0.0",
        "model_path": config.MODEL_PATH,
        "device": config.DEVICE,
        "max_length": config.MAX_LENGTH,
        "supported_sentiments": ["Negative", "Neutral", "Positive"],
        "features": [
            "Handles Nepali text",
            "Supports emojis",
            "Processes informal language",
            "Batch prediction support"
        ]
    }

# ============================================
# MAIN (for running directly)
# ============================================
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(
        "app:app",  # Change "app" to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True during development
        workers=1  # Increase for production
    )