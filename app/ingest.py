import os
from pypdf import PdfReader
import pandas as pd
from app.db import init_db, get_conn
from app.embeddings import embed
from tqdm import tqdm

DATA_DIR = "data"


def load_pdfs():
    texts = []
    for file in os.listdir(f"{DATA_DIR}/pdf"):
        if file.endswith(".pdf"):
            reader = PdfReader(f"{DATA_DIR}/pdf/{file}")
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts


def load_csvs():
    texts = []
    for file in os.listdir(f"{DATA_DIR}/csv"):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{DATA_DIR}/csv/{file}")
            for _, row in df.iterrows():
                texts.append(" ".join([f"{k}:{v}" for k, v in row.items()]))
    return texts


def main():
    init_db()
    texts = load_pdfs() + load_csvs()
    vectors = embed(texts)

    with get_conn() as conn:
        with conn.cursor() as cur:
            for text, vec in tqdm(zip(texts, vectors)):
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                    (text, vec.tolist())
                )
        conn.commit()


if __name__ == "__main__":
    main()
