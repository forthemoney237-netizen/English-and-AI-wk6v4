"""Flask API for GloVe word-vector arithmetic, hosted on Vercel."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

MODEL_NAME = "glove-wiki-gigaword-50"
MODEL_DIRECTORY = Path(__file__).resolve().parent / "model_data"

# Gensim reads this setting during import. It must match build_model.py so the
# deployed function loads the model that was packaged at build time.
os.environ["GENSIM_DATA_DIR"] = str(MODEL_DIRECTORY)

import gensim.downloader as gd  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402


app = Flask(__name__)
_model: Any | None = None
_model_lock = threading.Lock()


def get_model() -> Any:
    """Load the model which Vercel packaged during the deployment build."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            if not (MODEL_DIRECTORY / MODEL_NAME).is_dir():
                raise RuntimeError("The packaged GloVe model is missing. Redeploy the project.")
            _model = gd.load(MODEL_NAME)
    return _model


def terms_from(value: Any, field_name: str) -> list[str]:
    """Validate a list of model tokens sent by the browser."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of words.")
    if len(value) > 12:
        raise ValueError(f"{field_name} can contain at most 12 words.")

    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Each {field_name} entry must be text.")
        word = item.strip().lower()
        if not word:
            continue
        if len(word) > 64:
            raise ValueError("Words must be 64 characters or fewer.")
        cleaned.append(word)
    return cleaned


@app.get("/")
@app.get("/api")
@app.get("/api/")
def health() -> Any:
    return jsonify(
        {"ok": True, "model": MODEL_NAME, "modelLoaded": _model is not None}
    )


@app.post("/")
@app.post("/api")
@app.post("/api/")
def calculate() -> Any:
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Send a JSON object with positive and negative word lists.")

        positive = terms_from(payload.get("positive"), "positive")
        negative = terms_from(payload.get("negative"), "negative")
        if not positive and not negative:
            raise ValueError("Add at least one word.")

        requested_top_n = payload.get("topN", 5)
        if isinstance(requested_top_n, bool) or not isinstance(requested_top_n, int):
            raise ValueError("topN must be a whole number.")
        top_n = min(max(requested_top_n, 1), 10)

        model = get_model()
        unknown = [word for word in [*positive, *negative] if word not in model.key_to_index]
        if unknown:
            return jsonify(
                {
                    "error": "Some words are not in this vocabulary.",
                    "unknown": unknown,
                }
            ), 422

        matches = model.most_similar(positive=positive, negative=negative, topn=top_n)
        return jsonify(
            {
                "model": MODEL_NAME,
                "positive": positive,
                "negative": negative,
                "results": [
                    {"word": word, "similarity": round(float(score), 4)}
                    for word, score in matches
                ],
            }
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        return jsonify(
            {
                "error": "The GloVe model could not be loaded. Redeploy the project and try again."
            }
        ), 503
