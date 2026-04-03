from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import numpy as np
import re

app = Flask(__name__)

print("Loading ML model...\n")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def get_sentence_lengths(text):
    sentences = re.split(r'[.!?]', text)
    return [len(s.split()) for s in sentences if len(s.strip()) > 0]


def calculate_burstiness(text):
    lengths = get_sentence_lengths(text)
    if len(lengths) < 2:
        return 0
    return round(float(np.std(lengths)), 2)


def calculate_variance(text):
    lengths = get_sentence_lengths(text)
    if len(lengths) < 2:
        return 0
    return round(float(np.var(lengths)), 2)


def calculate_perplexity(text):
    words = text.split()
    unique_words = len(set(words))

    if len(words) == 0:
        return 0

    score = len(words) / unique_words
    return round(score * 10, 2)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    text = data.get('text', '')

    if text.strip() == "":
        return jsonify({"error": "Empty input"})

    
    result = classifier(
        text,
        candidate_labels=["AI-generated", "Human-written"]
    )

    labels = result['labels']
    scores = result['scores']

    predictions = []
    for i in range(len(labels)):
        predictions.append({
            "label": labels[i],
            "score": round(scores[i] * 100, 2)
        })

  
    perplexity = calculate_perplexity(text)
    burstiness = calculate_burstiness(text)
    variance = calculate_variance(text)

    return jsonify({
        "predictions": predictions,
        "final_verdict": labels[0],
        "perplexity": perplexity,
        "burstiness": burstiness,
        "variance": variance
    })


if __name__ == "__main__":
    app.run(debug=True)