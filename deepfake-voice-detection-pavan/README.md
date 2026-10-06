# Deepfake Voice Detection - Pavan

A Flask-based starter project for detecting whether an uploaded voice sample is likely human or AI-generated. The project is designed as a simple portfolio-ready prototype for audio fraud detection, misinformation prevention, and identity security.

## Features

- Upload `.wav` or `.mp3` audio files
- Extract basic audio features with `librosa`
- Run a lightweight prediction flow
- Store prediction history in SQLite
- Clean HTML/CSS interface

## Technologies Used

Python, Flask, Machine Learning, SQLite, HTML, CSS

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Note

This repository contains original starter code for demonstration and learning. For real-world use, train the model on a verified deepfake audio dataset and validate it carefully.
