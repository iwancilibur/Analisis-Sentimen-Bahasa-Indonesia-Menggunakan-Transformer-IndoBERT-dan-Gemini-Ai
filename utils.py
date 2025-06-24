from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import Config

def load_models(use_fine_tuned=False):
    """Muat model BERT (base atau fine-tuned)"""
    model_path = Config.MODEL_PATH if use_fine_tuned else Config.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Tambahkan token informal
    informal_tokens = ["banget", "gak", "ga", "nggak", "enggak", "tdk", "jgn"]
    tokenizer.add_tokens(informal_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    """Prediksi sentimen dengan confidence score"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    
    return pred.item(), confidence.item()