from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load model once
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route("/")
def home():
    return "Model is running."

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=200)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"summary": summary})
