import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    DATA_PATH = "./data"
    MODEL_PATH = "./models/fine_tuned_model"
    CONFIDENCE_THRESHOLD = 0.5