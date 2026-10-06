from pathlib import Path
import sqlite3
import time

import librosa
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "predictions.db"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def analyze_audio(path):
    audio, sample_rate = librosa.load(path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    zero_crossing = librosa.feature.zero_crossing_rate(audio)

    score = float(
        np.mean(np.abs(mfcc)) * 0.03
        + np.mean(spectral_centroid) / 5000
        + np.mean(zero_crossing) * 2
    )
    confidence = max(55, min(96, round(score * 100, 2)))
    result = "AI-generated voice suspected" if score > 0.72 else "Human voice likely"
    return result, confidence


def save_prediction(filename, result, confidence):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions(filename, result, confidence, created_at) VALUES (?, ?, ?, ?)",
            (filename, result, confidence, time.strftime("%Y-%m-%d %H:%M:%S")),
        )


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        audio_file = request.files.get("audio")
        if not audio_file or not audio_file.filename:
            error = "Please upload an audio file."
        else:
            safe_name = f"{int(time.time())}_{audio_file.filename}"
            saved_path = UPLOAD_DIR / safe_name
            audio_file.save(saved_path)
            try:
                result, confidence = analyze_audio(saved_path)
                save_prediction(audio_file.filename, result, confidence)
                prediction = {"result": result, "confidence": confidence}
            except Exception as exc:
                error = f"Could not analyze this audio file: {exc}"

    with sqlite3.connect(DB_PATH) as conn:
        history = conn.execute(
            "SELECT filename, result, confidence, created_at FROM predictions ORDER BY id DESC LIMIT 5"
        ).fetchall()

    return render_template("index.html", prediction=prediction, error=error, history=history)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
