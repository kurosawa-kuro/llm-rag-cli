import os
from pypdf import PdfReader
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.db import init_db, get_vectorstore
from app.chunking import split_by_structure
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

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

    csv_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = []

    # PDF: split_by_structure（段落ベース分割）
    for text, source in pdf_items:
        chunks = split_by_structure(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": source, "chunk_index": i},
            ))

    # CSV: RecursiveCharacterTextSplitter
    for text, source in csv_items:
        split_docs = csv_splitter.create_documents(
            [text],
            metadatas=[{"source": source}],
        )
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_index"] = i
            documents.append(doc)

    if documents:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents)


if __name__ == "__main__":
    main()
