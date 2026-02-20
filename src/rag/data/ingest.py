import os
from pypdf import PdfReader
import pandas as pd
from langchain_core.documents import Document
from rag.core.container import get_container
from rag.data.chunking import split_by_structure
from rag.core.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.infra.db import create_vectorstore

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
                content_parts = []
                for k, v in row.items():
                    if k.lower() not in ("category", "カテゴリ"):
                        content_parts.append(str(v))
                text = "\n".join(content_parts)
                texts.append((text, f"{file}:r{idx+1}"))
    return texts


def main():
    container = get_container()

    # 既存ドキュメントをクリア（コレクション削除→vectorstore再生成）
    container.vectorstore.delete_collection()
    container._vectorstore = create_vectorstore(container.embeddings)

    pdf_items = load_pdfs()
    csv_items = load_csvs()

    documents = []

    # PDF: split_by_structure（段落ベース分割）
    for text, source in pdf_items:
        chunks = split_by_structure(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": source, "chunk_index": i},
            ))

    # CSV: 1行=1ドキュメント（分割なし）
    for text, source in csv_items:
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": source, "chunk_index": 0},
            ))

    if documents:
        container.vectorstore.add_documents(documents)


if __name__ == "__main__":
    main()
