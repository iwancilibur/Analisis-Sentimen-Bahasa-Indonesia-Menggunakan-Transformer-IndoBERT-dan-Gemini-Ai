from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import torch
from utils import load_models
from config import Config
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_csv(file_path):
    """Validasi struktur file CSV"""
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) < 2:
            raise ValueError("File CSV harus memiliki minimal 2 kolom (teks dan label)")
        logger.info("\nContoh 5 baris pertama:")
        logger.info(df.head())
        return True
    except Exception as e:
        logger.error(f"Error validasi CSV: {e}")
        return False

def load_dataset(file_path):
    """Load dataset dengan penanganan error lebih baik"""
    try:
        df = pd.read_csv(file_path)
        
        # Cek kolom yang tersedia
        available_columns = df.columns.tolist()
        logger.info(f"Kolom yang tersedia: {available_columns}")
        
        # Cari kolom teks (case insensitive)
        text_col = None
        possible_text_columns = ['text', 'teks', 'kalimat', 'review', 'content']
        for col in possible_text_columns:
            if col.lower() in [c.lower() for c in df.columns]:
                text_col = col
                break
                
        if not text_col:
            raise ValueError(f"Kolom teks tidak ditemukan. Kolom yang ada: {df.columns.tolist()}")
            
        # Cari kolom label
        label_col = None
        possible_label_columns = ['label', 'sentimen', 'sentiment', 'class']
        for col in possible_label_columns:
            if col.lower() in [c.lower() for c in df.columns]:
                label_col = col
                break
                
        if not label_col:
            label_col = df.columns[1]  # Fallback ke kolom kedua
            
        logger.info(f"Menggunakan kolom: '{text_col}' untuk teks dan '{label_col}' untuk label")
        
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
        
        return texts, labels
        
    except Exception as e:
        logger.error(f"Gagal memuat dataset: {str(e)}", exc_info=True)
        raise

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }
    
    def __len__(self):
        return len(self.labels)

def fine_tune():
    try:
        # Validasi CSV
        if not validate_csv(f"{Config.DATA_PATH}/train.csv"):
            raise ValueError("Format train.csv tidak valid")
            
        if not validate_csv(f"{Config.DATA_PATH}/test.csv"):
            raise ValueError("Format test.csv tidak valid")

        # Cek GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ Device: {device}")

        # Load model
        tokenizer, model = load_models()
        model.to(device)

        # Load data
        logger.info("Memuat dataset...")
        train_texts, train_labels = load_dataset(f"{Config.DATA_PATH}/train.csv")
        val_texts, val_labels = load_dataset(f"{Config.DATA_PATH}/test.csv")
        
        logger.info(f"Jumlah data latih: {len(train_texts)}")
        logger.info(f"Jumlah data validasi: {len(val_texts)}")

        # Dataset
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            report_to="none",
            save_total_limit=2
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Mulai training
        logger.info("🚀 Memulai training...")
        trainer.train()
        
        # Simpan model
        model.save_pretrained(Config.MODEL_PATH)
        tokenizer.save_pretrained(Config.MODEL_PATH)
        logger.info(f"💾 Model disimpan di: {Config.MODEL_PATH}")

    except Exception as e:
        logger.error("❌ Gagal melakukan fine-tuning", exc_info=True)
        raise

if __name__ == "__main__":
    print("="*50)
    print("🔍 Fine-Tuning Model Dimulai")
    print("="*50)
    try:
        fine_tune()
    except Exception as e:
        logger.error("❌ Program dihentikan karena error")
        sys.exit(1)