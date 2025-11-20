"""
Optimized Nepali Sentiment Analysis - mBERT Version
===================================================
Using bert-base-multilingual-cased with optimized settings
"""

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import random
from torch.optim import AdamW
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================
# CONFIGURATION (OPTIMIZED FOR mBERT)
# ============================================
class Config:
    # Dataset
    DATASET_NAME = "Shushant/NepaliSentiment"
    
    # --- USING mBERT ---
    MODEL_NAME = "bert-base-multilingual-cased"
    NUM_LABELS = 3
    
    # --- TUNED FOR mBERT ---
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 8  # mBERT might need slightly fewer epochs
    LEARNING_RATE = 2e-5  # Slightly higher LR for BERT
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_LENGTH = 256
    
    # Advanced settings
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0
    LR_SCHEDULER_TYPE = "cosine"
    
    # Data split
    VAL_SPLIT = 0.15
    RANDOM_SEED = 42
    
    # Directories
    OUTPUT_DIR = "./results_mbert_optimized"
    SAVED_MODEL_DIR = "./nepali_sentiment_mbert_model"
    PLOTS_DIR = "./plots_mbert"
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# ============================================
# SETUP
# ============================================
def setup_environment():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    print(f"🚀 NEPALI SENTIMENT ANALYSIS - mBERT OPTIMIZED")
    print(f"📊 Model: {config.MODEL_NAME}")
    print(f"⚡ Device: {config.DEVICE}")

# ============================================
# DATA LOADING FROM HUGGING FACE
# ============================================
def load_and_prepare_data():
    """Load dataset from Hugging Face and prepare for training"""
    print("📥 Loading dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset(config.DATASET_NAME)
    
    # Convert to pandas for easier manipulation
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    print(f"📊 Training samples: {len(train_df)}")
    print(f"📊 Test samples: {len(test_df)}")
    print(f"📊 Label distribution in training: {train_df['label'].value_counts().sort_index().to_dict()}")
    
    # Clean and preprocess
    for df in [train_df, test_df]:
        df['text'] = df['text'].fillna('').astype(str).str.strip()
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df.dropna(subset=['label'], inplace=True)
        df['label'] = df['label'].astype(int)
        
        # Enhanced Nepali text cleaning
        df['text'] = df['text'].apply(clean_nepali_text)
    
    # Filter valid labels
    train_df = train_df[train_df['label'].isin([0, 1, 2])]
    test_df = test_df[test_df['label'].isin([0, 1, 2])]
    
    return train_df, test_df

def clean_nepali_text(text):
    """Enhanced cleaning for Nepali text"""
    # Remove zero-width characters
    text = text.replace('\u200d', '').replace('\u200c', '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def create_validation_set(train_df):
    """Create validation set with stratification"""
    train_split, val_split = train_test_split(
        train_df, 
        test_size=config.VAL_SPLIT, 
        stratify=train_df['label'],
        random_state=config.RANDOM_SEED
    )
    
    print(f"📊 Training samples after split: {len(train_split)}")
    print(f"📊 Validation samples: {len(val_split)}")
    
    return train_split, val_split

# ============================================
# TOKENIZATION & DATASET
# ============================================
def process_data(train_df, val_df, test_df):
    """Tokenize and prepare datasets"""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )

    datasets = {
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    }

    tokenized_datasets = {}
    for name, ds in datasets.items():
        tokenized = ds.map(tokenize_function, batched=True)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_datasets[name] = tokenized
        
    return tokenized_datasets, tokenizer

# ============================================
# ADVANCED TRAINING
# ============================================
class AdvancedWeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=weights, 
                label_smoothing=config.LABEL_SMOOTHING
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss(
                label_smoothing=config.LABEL_SMOOTHING
            )
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro'),
        'weighted_f1': f1_score(labels, preds, average='weighted'),
        'macro_precision': precision_score(labels, preds, average='macro'),
        'macro_recall': recall_score(labels, preds, average='macro')
    }

def main():
    setup_environment()
    
    # 1. Load Data from Hugging Face
    train_df, test_df = load_and_prepare_data()
    train_split, val_split = create_validation_set(train_df)
    
    # 2. Calculate Class Weights
    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(train_split['label']), 
        y=train_split['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"⚖️ Class Weights: {class_weights}")

    # 3. Tokenize Data
    tokenized_datasets, tokenizer = process_data(train_split, val_split, test_df)
    
    # 4. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=config.NUM_LABELS
    )
    model.to(config.DEVICE)

    # 5. Training Arguments (Optimized for mBERT)
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        num_train_epochs=config.EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        logging_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=config.MAX_GRAD_NORM,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # 6. Train
    trainer = AdvancedWeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)]
    )

    print("\n🏋️ Starting Training with mBERT...")
    train_result = trainer.train()

    # 7. Comprehensive Evaluation
    print("\n📊 Evaluating on Test Set...")
    test_predictions = trainer.predict(tokenized_datasets["test"])
    test_metrics = compute_metrics(test_predictions)
    
    print(f"🏆 Final Test Results:")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"   Weighted F1: {test_metrics['weighted_f1']:.4f}")
    
    # Detailed classification report
    preds = np.argmax(test_predictions.predictions, axis=1)
    print("\n📋 Detailed Classification Report:")
    print(classification_report(test_df['label'], preds, 
                              target_names=['Negative', 'Neutral', 'Positive']))

    # Save the model
    trainer.save_model(config.SAVED_MODEL_DIR)
    tokenizer.save_pretrained(config.SAVED_MODEL_DIR)
    print(f"✅ Model saved to {config.SAVED_MODEL_DIR}")

if __name__ == "__main__":
    main()