import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast

# --- Configuration ---
# Your DevOps pipeline must place the model artifacts inside this directory
MODEL_DIR = "./model"

app = FastAPI(
    title="🛡️ DistilBERT Phishing Detection Endpoint (ONNX Optimized)", 
    version="1.0"
)

# Global memory handles for the container
tokenizer = None
ort_session = None

def softmax(x):
    """Computes categorical probability distribution using pure NumPy."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.on_event("startup")
async def startup_event():
    global tokenizer, ort_session
    
    # Verify the DevOps pipeline has properly provisioned the weights
    onnx_path = os.path.join(MODEL_DIR, "model.onnx")
    if not os.path.exists(onnx_path):
        raise RuntimeError(
            f"Critical Error: 'model.onnx' not found at {onnx_path}. "
            "Ensure your DevOps pipeline maps or injects the model folder correctly."
        )
    
    print("🧠 Initializing ONNX Inference Runtime Session...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    ort_session = ort.InferenceSession(onnx_path)
    print("🔥 API is officially live and routing requests.")

# --- API Data Schemas via Pydantic ---
class EmailPayload(BaseModel):
    text: str

# --- Endpoints ---
@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "engine": "onnxruntime-cpu", 
        "target_model": "DistilBERT-v1"
    }

@app.post("/predict")
def predict(payload: EmailPayload):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
        
    # 1. Tokenize text input directly into NumPy arrays
    inputs = tokenizer(
        payload.text, 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="np"
    )
    
    # 2. Map input tensor types to match ONNX expected schema
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    # 3. Fire highly optimized execution graph
    ort_outputs = ort_session.run(None, ort_inputs)
    logits = ort_outputs[0]
    
    # 4. Map output raw logits into discrete percentages
    probabilities = softmax(logits)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence_score = float(probabilities[predicted_class])
    
    label_map = {0: "Safe", 1: "Phishing"}
    
    return {
        "prediction": label_map[predicted_class],
        "confidence": round(confidence_score * 100, 2),
        "class_id": predicted_class
    }
