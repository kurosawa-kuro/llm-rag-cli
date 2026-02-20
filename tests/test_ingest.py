from unittest.mock import patch, MagicMock, call
from langchain_core.documents import Document
import numpy as np
import pytest


class TestLoadPdfs:
    @patch("app.ingest.PdfReader")
    @patch("app.ingest.os.listdir", return_value=["doc.pdf", "notes.txt"])
    def test_loads_only_pdf_files(self, mock_listdir, mock_reader):
        page = MagicMock()
        page.extract_text.return_value = "page text"
        mock_reader.return_value.pages = [page]
        from app.ingest import load_pdfs

        result = load_pdfs()

        mock_reader.assert_called_once_with("data/pdf/doc.pdf")
        assert result == [("page text", "doc.pdf:p1")]

    @patch("app.ingest.PdfReader")
    @patch("app.ingest.os.listdir", return_value=["a.pdf"])
    def test_extracts_all_pages(self, mock_listdir, mock_reader):
        page1 = MagicMock()
        page1.extract_text.return_value = "page 1"
        page2 = MagicMock()
        page2.extract_text.return_value = "page 2"
        mock_reader.return_value.pages = [page1, page2]
        from app.ingest import load_pdfs

        result = load_pdfs()
        assert result == [("page 1", "a.pdf:p1"), ("page 2", "a.pdf:p2")]

    @patch("app.ingest.PdfReader")
    @patch("app.ingest.os.listdir", return_value=["first.pdf", "second.pdf"])
    def test_loads_multiple_pdf_files(self, mock_listdir, mock_reader):
        page_a = MagicMock()
        page_a.extract_text.return_value = "content A"
        page_b = MagicMock()
        page_b.extract_text.return_value = "content B"

        mock_reader.side_effect = [
            MagicMock(pages=[page_a]),
            MagicMock(pages=[page_b]),
        ]
        from app.ingest import load_pdfs

        result = load_pdfs()
        assert len(result) == 2
        assert result[0] == ("content A", "first.pdf:p1")
        assert result[1] == ("content B", "second.pdf:p1")

    @patch("app.ingest.PdfReader")
    @patch("app.ingest.os.listdir", return_value=["a.pdf"])
    def test_returns_tuple_of_text_and_source(self, mock_listdir, mock_reader):
        page = MagicMock()
        page.extract_text.return_value = "text"
        mock_reader.return_value.pages = [page]
        from app.ingest import load_pdfs

        result = load_pdfs()
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        text, source = result[0]
        assert isinstance(text, str)
        assert isinstance(source, str)

    @patch("app.ingest.PdfReader")
    @patch("app.ingest.os.listdir", return_value=["doc.pdf"])
    def test_page_numbering_starts_at_1(self, mock_listdir, mock_reader):
        pages = [MagicMock() for _ in range(3)]
        for i, p in enumerate(pages):
            p.extract_text.return_value = f"page {i}"
        mock_reader.return_value.pages = pages
        from app.ingest import load_pdfs

        result = load_pdfs()
        assert result[0][1] == "doc.pdf:p1"
        assert result[1][1] == "doc.pdf:p2"
        assert result[2][1] == "doc.pdf:p3"


class TestLoadCsvs:
    @patch("app.ingest.pd.read_csv")
    @patch("app.ingest.os.listdir", return_value=["data.csv", "readme.md"])
    def test_loads_only_csv_files(self, mock_listdir, mock_read_csv):
        import pandas as pd
        mock_read_csv.return_value = pd.DataFrame({"name": ["Alice"], "age": [30]})
        from app.ingest import load_csvs

        result = load_csvs()

        mock_read_csv.assert_called_once_with("data/csv/data.csv")
        assert len(result) == 1
        assert result[0][0] == "name:Alice age:30"
        assert result[0][1] == "data.csv:r1"

    @patch("app.ingest.pd.read_csv")
    @patch("app.ingest.os.listdir", return_value=["data.csv"])
    def test_converts_rows_to_key_value(self, mock_listdir, mock_read_csv):
        import pandas as pd
        mock_read_csv.return_value = pd.DataFrame({
            "col1": ["a", "b"],
            "col2": [1, 2],
        })
        from app.ingest import load_csvs

        result = load_csvs()
        assert len(result) == 2
        assert "col1:a" in result[0][0]
        assert "col2:1" in result[0][0]
        assert result[0][1] == "data.csv:r1"
        assert result[1][1] == "data.csv:r2"

    @patch("app.ingest.pd.read_csv")
    @patch("app.ingest.os.listdir", return_value=["faq.csv", "products.csv"])
    def test_loads_multiple_csv_files(self, mock_listdir, mock_read_csv):
        import pandas as pd
        mock_read_csv.side_effect = [
            pd.DataFrame({"q": ["q1"]}),
            pd.DataFrame({"name": ["p1"]}),
        ]
        from app.ingest import load_csvs

        result = load_csvs()
        assert len(result) == 2
        assert result[0][1] == "faq.csv:r1"
        assert result[1][1] == "products.csv:r1"


class TestMain:
    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[("csv text", "file.csv:r1")])
    @patch("app.ingest.load_pdfs", return_value=[("pdf text", "doc.pdf:p1")])
    @patch("app.ingest.init_db")
    def test_full_pipeline(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        mock_init.assert_called_once()
        mock_pdfs.assert_called_once()
        mock_csvs.assert_called_once()
        mock_vs.return_value.add_documents.assert_called_once()
        docs = mock_vs.return_value.add_documents.call_args[0][0]
        assert len(docs) == 2

    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[("csv text", "file.csv:r1")])
    @patch("app.ingest.load_pdfs", return_value=[("pdf text", "doc.pdf:p1")])
    @patch("app.ingest.init_db")
    def test_documents_have_metadata(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        docs = mock_vs.return_value.add_documents.call_args[0][0]
        for doc in docs:
            assert isinstance(doc, Document)
            assert "source" in doc.metadata
            assert "chunk_index" in doc.metadata

    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[])
    @patch("app.ingest.load_pdfs", return_value=[("para1\n\npara2\n\npara3", "doc.pdf:p1")])
    @patch("app.ingest.init_db")
    def test_pdf_uses_split_by_structure(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        docs = mock_vs.return_value.add_documents.call_args[0][0]
        assert len(docs) == 3  # 3 paragraphs

    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[("csv row text", "data.csv:r1")])
    @patch("app.ingest.load_pdfs", return_value=[])
    @patch("app.ingest.init_db")
    def test_csv_uses_text_splitter(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        docs = mock_vs.return_value.add_documents.call_args[0][0]
        # Short CSV text = 1 chunk
        assert len(docs) == 1

    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[])
    @patch("app.ingest.load_pdfs", return_value=[])
    @patch("app.ingest.init_db")
    def test_main_with_no_documents(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        mock_init.assert_called_once()
        mock_vs.return_value.add_documents.assert_not_called()

    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[
        ("row1 text", "data.csv:r1"),
        ("row2 text", "data.csv:r2"),
    ])
    @patch("app.ingest.load_pdfs", return_value=[
        ("pdf text", "doc.pdf:p1"),
    ])
    @patch("app.ingest.init_db")
    def test_main_adds_all_documents_at_once(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        mock_vs.return_value.add_documents.assert_called_once()
        docs = mock_vs.return_value.add_documents.call_args[0][0]
        assert len(docs) == 3

    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[("csv text", "file.csv:r1")])
    @patch("app.ingest.load_pdfs", return_value=[("pdf text", "doc.pdf:p1")])
    @patch("app.ingest.init_db")
    def test_main_creates_correct_documents(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        docs = mock_vs.return_value.add_documents.call_args[0][0]
        for doc in docs:
            assert isinstance(doc, Document)
            assert isinstance(doc.page_content, str)
            assert isinstance(doc.metadata["source"], str)
            assert isinstance(doc.metadata["chunk_index"], int)

    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[])
    @patch("app.ingest.load_pdfs", return_value=[("long " * 200, "doc.pdf:p1")])
    @patch("app.ingest.init_db")
    def test_long_text_produces_multiple_chunks(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        docs = mock_vs.return_value.add_documents.call_args[0][0]
        assert len(docs) > 1


class TestLoadEdgeCases:
    @patch("app.ingest.os.listdir", return_value=[])
    def test_load_pdfs_empty_directory(self, mock_listdir):
        from app.ingest import load_pdfs

        result = load_pdfs()
        assert result == []

    @patch("app.ingest.os.listdir", return_value=[])
    def test_load_csvs_empty_directory(self, mock_listdir):
        from app.ingest import load_csvs

        result = load_csvs()
        assert result == []

    @patch("app.ingest.PdfReader")
    @patch("app.ingest.os.listdir", return_value=["doc.pdf"])
    def test_load_pdfs_empty_page_text(self, mock_listdir, mock_reader):
        page = MagicMock()
        page.extract_text.return_value = ""
        mock_reader.return_value.pages = [page]
        from app.ingest import load_pdfs

        result = load_pdfs()
        # 空テキストでもタプルとして返される
        assert result == [("", "doc.pdf:p1")]

    @patch("app.ingest.PdfReader")
    @patch("app.ingest.os.listdir", return_value=["doc.pdf"])
    def test_load_pdfs_none_text_from_page(self, mock_listdir, mock_reader):
        page = MagicMock()
        page.extract_text.return_value = None
        mock_reader.return_value.pages = [page]
        from app.ingest import load_pdfs

        result = load_pdfs()
        assert result == [(None, "doc.pdf:p1")]

    @patch("app.ingest.os.listdir", side_effect=FileNotFoundError)
    def test_load_pdfs_missing_directory_raises_error(self, mock_listdir):
        from app.ingest import load_pdfs

        with pytest.raises(FileNotFoundError):
            load_pdfs()

    @patch("app.ingest.os.listdir", side_effect=FileNotFoundError)
    def test_load_csvs_missing_directory_raises_error(self, mock_listdir):
        from app.ingest import load_csvs

        with pytest.raises(FileNotFoundError):
            load_csvs()


class TestMainEdgeCases:
    @patch("app.ingest.get_vectorstore")
    @patch("app.ingest.load_csvs", return_value=[("", "empty.csv:r1")])
    @patch("app.ingest.load_pdfs", return_value=[])
    @patch("app.ingest.init_db")
    def test_empty_csv_text_produces_no_documents(self, mock_init, mock_pdfs, mock_csvs, mock_vs):
        from app.ingest import main

        main()

        # 空テキストはTextSplitterがドキュメントを生成しないため add_documents 未呼出
        mock_vs.return_value.add_documents.assert_not_called()
