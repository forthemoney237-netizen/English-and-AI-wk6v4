"""Download the GloVe data once during the Vercel build.

Vercel installs Python dependencies before running this script.  The downloaded
files are then included in the Flask function bundle, avoiding a download while
a visitor waits for a result.
"""

import os
from pathlib import Path


MODEL_NAME = "glove-wiki-gigaword-50"
MODEL_DIRECTORY = Path(__file__).resolve().parent / "model_data"

# Gensim reads this setting when it is imported.  Set it first so its downloader
# uses the project folder rather than the read-only Vercel home directory.
os.environ["GENSIM_DATA_DIR"] = str(MODEL_DIRECTORY)

import gensim.downloader as gd  # noqa: E402


def main() -> None:
    MODEL_DIRECTORY.mkdir(exist_ok=True)
    gd.load(MODEL_NAME)
    print(f"{MODEL_NAME} is ready in {MODEL_DIRECTORY}")


if __name__ == "__main__":
    main()
