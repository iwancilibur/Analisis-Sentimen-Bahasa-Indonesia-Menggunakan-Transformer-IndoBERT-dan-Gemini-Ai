from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from config import Config
from utils import load_models, predict_sentiment
from fine_tuning import fine_tune
from evaluate import evaluate
import time

app = Flask(__name__)

# Load model
tokenizer, model = load_models(use_fine_tuned=True)

# Gemini AI
genai.configure(api_key=Config.GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Masukkan teks terlebih dahulu!"})
    
    try:
        # Prediksi sentimen
        pred, confidence = predict_sentiment(text, tokenizer, model)
        sentiment = ["Positif", "Netral", "Negatif"][pred]
        
        # Penjelasan Gemini
        prompt = f"""
        [Instruksi]
        Berikan analisis untuk teks berikut dengan ketentuan:
        1. Hasil analisis sentimen: {sentiment} (Tingkat Kepercayaan: {confidence:.0%})
        2. Teks: "{text}"
        
        Format respons:
        - Analisis: [Jelaskan mengapa teks dikategorikan seperti hasil analisis]
        - Catatan: [Berikan catatan jika ada kemungkinan ketidaksesuaian]
        - Saran: [Berikan saran terkait sentimen yang terdeteksi]
        
        Gunakan Bahasa Indonesia yang formal dan jelas. Maksimal 5 kalimat.
        """
        ai_response = gemini.generate_content(prompt).text
        
        return jsonify({
            "sentiment": sentiment,
            "confidence": f"{confidence:.0%}",
            "ai_response": ai_response,
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/train", methods=["POST"])
def train():
    try:
        fine_tune()
        metrics = evaluate()
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)