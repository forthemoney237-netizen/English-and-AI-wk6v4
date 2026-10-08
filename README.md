# Word Vector Lab

A small Vercel app that performs word-vector arithmetic with the same model
used in the reference notebook:

```python
import gensim.downloader as gd
glove = gd.load("glove-wiki-gigaword-50")
```

Enter additions and subtractions as comma-separated words. For example,
`king, woman` in **Add words** and `man` in **Remove words** asks for the words
closest to `king + woman - man`.

## Project layout

- `public/` — a dependency-free browser interface served by Vercel.
- `app.py` — Flask API deployed as the Python Vercel Function, with its
  calculator endpoint at `/api`.
- `build_model.py` — downloads the GloVe files while Vercel builds the
  deployment, so a visitor does not wait for a model download.
- `requirements.txt` — Python packages used by the function.

## Run locally

Use Python 3.12, create an environment, install the requirements, then start
the Flask app:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python build_model.py
flask --app app:app run
```

Open the static page through the Vercel development server, or serve `public/`
from another local static server and send requests to the Flask endpoint.

## Deploy to Vercel

1. Create a new GitHub repository and upload this project (not its parent
   folder).
2. In Vercel, choose **New Project**, import that repository, and click
   **Deploy**. No environment variables are needed.
3. Open the generated site. Vercel downloads the GloVe data during the build,
   then packages it with the Flask function.

Vercel detects the Flask application in `app.py` and serves `public/` as static
content. The application exposes its calculator at `/api`.

## Production note

This is a good demonstration app. For high traffic, consider object storage or
a dedicated inference service so the model can be shared across instances.
