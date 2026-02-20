import os
from pypdf import PdfReader
import pandas as pd
from app.db import init_db, get_conn
from app.embeddings import embed
from app.chunking import split_text, split_by_structure
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from tqdm import tqdm

DATA_DIR = "data"


def load_pdfs():
    texts = []
    for file in os.listdir(f"{DATA_DIR}/pdf"):
        if file.endswith(".pdf"):
            reader = PdfReader(f"{DATA_DIR}/pdf/{file}")
            for i, page in enumerate(reader.pages):
                texts.append((page.extract_text(), f"{file}:p{i+1}"))
    return texts


def load_csvs():
    texts = []
    for file in os.listdir(f"{DATA_DIR}/csv"):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{DATA_DIR}/csv/{file}")
            for idx, row in df.iterrows():
                text = " ".join([f"{k}:{v}" for k, v in row.items()])
                texts.append((text, f"{file}:r{idx+1}"))
    return texts


def main():
    init_db()
    pdf_items = load_pdfs()
    csv_items = load_csvs()

    # チャンク分割: PDFは構造Chunk、CSVは固定サイズ
    chunks_with_meta = []
    for text, source in pdf_items:
        chunks = split_by_structure(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunks_with_meta.append((chunk, source, i))
    for text, source in csv_items:
        chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunks_with_meta.append((chunk, source, i))

    texts = [c[0] for c in chunks_with_meta]
    vectors = embed(texts)

    with get_conn() as conn:
        with conn.cursor() as cur:
            for (text, source, chunk_index), vec in tqdm(zip(chunks_with_meta, vectors)):
                cur.execute(
                    "INSERT INTO documents (content, embedding, source, chunk_index) VALUES (%s, %s, %s, %s)",
                    (text, vec.tolist(), source, chunk_index)
                )
        conn.commit()


if __name__ == "__main__":
    main()
