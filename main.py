import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer

# The Docker container will copy your local folder directly here
MODEL_DIR = "./model"

app = FastAPI(
    title="🛡️ DistilBERT Phishing Detection API (Ultra-Lightweight)", 
    version="1.0"
)

# Global memory handles
tokenizer = None
ort_session = None

def softmax(x):
    """Computes probabilities using pure NumPy."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.on_event("startup")
async def startup_event():
    global tokenizer, ort_session
    
    onnx_path = os.path.join(MODEL_DIR, "model.onnx")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    
    if not os.path.exists(onnx_path) or not os.path.exists(tokenizer_path):
        raise RuntimeError(
            f"Critical Error: Missing files in {MODEL_DIR}. "
            "Ensure 'model.onnx' and 'tokenizer.json' are present."
        )
    
    print("🧠 Initializing pure Rust Tokenizer and ONNX Runtime...")
    
    # 1. Load pure rust tokenizer (Bypassing Hugging Face Transformers)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 2. Enforce standard DistilBERT padding and truncation logic
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(length=512, pad_id=0, pad_token="[PAD]")
    
    # 3. Load ONNX execution graph
    ort_session = ort.InferenceSession(onnx_path)
    print("🔥 API is officially live and routing requests.")

# --- Data Schemas ---
class EmailPayload(BaseModel):
    text: str

# --- Endpoints ---
@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "engine": "onnxruntime-cpu & pure-tokenizers", 
        "target_model": "DistilBERT-v1"
    }

@app.post("/predict")
def predict(payload: EmailPayload):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
        
    # 1. Tokenize directly to numerical IDs using the Rust engine
    encoding = tokenizer.encode(payload.text)
    
    # 2. Package into 2D NumPy arrays expected by ONNX [batch_size, sequence_length]
    ort_inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64)
    }
    
    # 3. Fire highly optimized execution graph
    ort_outputs = ort_session.run(None, ort_inputs)
    logits = ort_outputs[0]
    
    # 4. Map output raw logits into discrete probabilities
    probabilities = softmax(logits)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence_score = float(probabilities[predicted_class])
    
    # 5. Extract specific probability strings
    safe_prob = float(probabilities[0])
    phishing_prob = float(probabilities[1])
    
    label_map = {0: "Safe", 1: "Phishing"}
    
    return {
        "prediction": label_map[predicted_class],
        "confidence": f"{round(confidence_score * 100, 2)}%",
        "probabilities": {
            "Safe": f"{round(safe_prob * 100, 2)}%",
            "Phishing": f"{round(phishing_prob * 100, 2)}%"
        },
        "class_id": predicted_class
    }
