import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from utils import load_models
from fine_tuning import SentimentDataset
from config import Config
from torch.utils.data import DataLoader
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """Load dataset dengan penanganan error"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset dimuat: {file_path}")
        logger.info(f"Contoh data:\n{df.head()}")
        return df["text"].tolist(), df["label"].tolist()
    except Exception as e:
        logger.error(f"Gagal memuat dataset: {e}")
        raise

def plot_confusion_matrix(y_true, y_pred, classes):
    """Visualisasi confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('./static/confusion_matrix.png')
    plt.close()
    logger.info("Confusion matrix disimpan di ./static/confusion_matrix.png")

def evaluate():
    try:
        # Cek GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ Menggunakan device: {device}")

        # Load model fine-tuned
        logger.info("Memuat model fine-tuned...")
        tokenizer, model = load_models(use_fine_tuned=True)
        model.to(device)

        # Load data test
        logger.info("Memuat dataset test...")
        test_texts, test_labels = load_dataset(f"{Config.DATA_PATH}/test.csv")

        # Buat dataset
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Prediksi
        logger.info("Memulai evaluasi...")
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        # Hitung metrik
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        precision_neg = precision_score(true_labels, predictions, labels=[2], average='micro')
        recall_pos = recall_score(true_labels, predictions, labels=[0], average='micro')

        # Classification report
        report = classification_report(
            true_labels,
            predictions,
            target_names=['Positif', 'Netral', 'Negatif'],
            digits=4
        )

        # Confusion matrix
        plot_confusion_matrix(true_labels, predictions, ['Positif', 'Netral', 'Negatif'])

        # Hasil
        metrics = {
            "Model": "Fine-Tuned IndoBERT",
            "Accuracy": f"{accuracy:.4f}",
            "F1-Score": f"{f1:.4f}",
            "Precision (Negatif)": f"{precision_neg:.4f}",
            "Recall (Positif)": f"{recall_pos:.4f}",
            "Classification Report": report
        }

        # Simpan hasil
        os.makedirs('./results', exist_ok=True)
        with open('./results/evaluation_results.txt', 'w') as f:
            for key, value in metrics.items():
                if key != "Classification Report":
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"\n{value}")

        logger.info("✅ Evaluasi selesai!")
        return metrics

    except Exception as e:
        logger.error(f"❌ Error dalam evaluasi: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    print("="*50)
    print("📊 Evaluasi Model Dimulai")
    print("="*50)
    
    try:
        results = evaluate()
        
        print("\nHasil Evaluasi:")
        print(f"Accuracy: {results['Accuracy']}")
        print(f"F1-Score: {results['F1-Score']}")
        print(f"Precision (Negatif): {results['Precision (Negatif)']}")
        print(f"Recall (Positif): {results['Recall (Positif)']}")
        print(f"\nClassification Report:\n{results['Classification Report']}")
        
    except Exception as e:
        logger.error("❌ Gagal melakukan evaluasi")
        exit(1)