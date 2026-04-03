# AiVsHumanDetector
AI vs Human Text Detector

This project is a web-based application that detects whether a given text is AI-generated or human-written using Machine Learning.

Features
Detect AI vs Human-written text
Confidence score with progress bars
Clean and modern dark UI
Real-time analysis using Flask backend
Additional text metrics (optional):
Perplexity
Burstiness
Sentence length variance
🛠️ Tech Stack
Frontend: HTML, CSS, JavaScript
Backend: Python (Flask)
ML Model: Hugging Face Transformers (facebook/bart-large-mnli)
Installation
Clone the repository:
git clone <your-repo-link>
cd ai-detector
Install dependencies:
pip install flask transformers torch
Run the app:
python app.py
Open in browser:
http://127.0.0.1:5000/
How it Works
User enters text in the UI
Frontend sends request to Flask API
ML model analyzes text
Results (AI/Human + confidence) are displayed
