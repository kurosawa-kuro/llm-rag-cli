from unittest.mock import patch, MagicMock, call
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
        assert result == ["page text"]

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
        assert result == ["page 1", "page 2"]


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
        assert "name:Alice" in result[0]
        assert "age:30" in result[0]

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
        assert "col1:a" in result[0]
        assert "col2:1" in result[0]


class TestMain:
    @patch("app.ingest.tqdm", side_effect=lambda x: x)
    @patch("app.ingest.get_conn")
    @patch("app.ingest.embed")
    @patch("app.ingest.load_csvs", return_value=["csv text"])
    @patch("app.ingest.load_pdfs", return_value=["pdf text"])
    @patch("app.ingest.init_db")
    def test_full_pipeline(self, mock_init, mock_pdfs, mock_csvs, mock_embed,
                           mock_conn, mock_tqdm, mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384, [0.2] * 384])
        from app.ingest import main

        main()

        mock_init.assert_called_once()
        mock_pdfs.assert_called_once()
        mock_csvs.assert_called_once()
        mock_embed.assert_called_once_with(["pdf text", "csv text"])
        assert cur.execute.call_count == 2
        conn.commit.assert_called_once()
