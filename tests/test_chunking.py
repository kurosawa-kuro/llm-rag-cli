class TestSplitText:
    def test_short_text_returns_single_chunk(self):
        from app.chunking import split_text

        result = split_text("short text", chunk_size=500)
        assert result == ["short text"]

    def test_empty_string_returns_empty_list(self):
        from app.chunking import split_text

        result = split_text("")
        assert result == []

    def test_long_text_splits_into_multiple_chunks(self):
        from app.chunking import split_text

        text = "word " * 200  # 1000 chars
        result = split_text(text, chunk_size=300, overlap=50)
        assert len(result) > 1

    def test_overlap_applied_correctly(self):
        from app.chunking import split_text

        text = "word " * 200
        result = split_text(text, chunk_size=300, overlap=50)
        # 2番目のチャンクの先頭部分が1番目の末尾と重複する
        for i in range(len(result) - 1):
            tail = result[i][-50:]
            assert tail in result[i + 1]

    def test_splits_at_word_boundary(self):
        from app.chunking import split_text

        text = "hello world this is a test sentence for chunking"
        result = split_text(text, chunk_size=20, overlap=5)
        for chunk in result:
            # チャンクが単語の途中で切れていないことを確認
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_no_data_loss(self):
        from app.chunking import split_text

        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        result = split_text(text, chunk_size=30, overlap=10)
        all_words = set(text.split())
        chunk_words = set()
        for chunk in result:
            chunk_words.update(chunk.split())
        assert all_words == chunk_words

    def test_different_chunk_sizes_produce_different_counts(self):
        from app.chunking import split_text

        text = "word " * 200
        result_small = split_text(text, chunk_size=300, overlap=50)
        result_large = split_text(text, chunk_size=800, overlap=50)
        assert len(result_small) > len(result_large)


class TestSplitByStructure:
    def test_splits_by_double_newline(self):
        from app.chunking import split_by_structure

        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        result = split_by_structure(text)
        assert result == ["paragraph one", "paragraph two", "paragraph three"]

    def test_single_paragraph_returns_single_chunk(self):
        from app.chunking import split_by_structure

        result = split_by_structure("just one paragraph")
        assert result == ["just one paragraph"]

    def test_empty_input_returns_empty_list(self):
        from app.chunking import split_by_structure

        result = split_by_structure("")
        assert result == []

    def test_strips_whitespace_from_paragraphs(self):
        from app.chunking import split_by_structure

        text = "  para one  \n\n  para two  "
        result = split_by_structure(text)
        assert result == ["para one", "para two"]

    def test_skips_empty_paragraphs(self):
        from app.chunking import split_by_structure

        text = "para one\n\n\n\n\npara two"
        result = split_by_structure(text)
        assert result == ["para one", "para two"]

    def test_long_paragraph_split_further_with_chunk_size(self):
        from app.chunking import split_by_structure

        long_para = "word " * 200
        text = f"short para\n\n{long_para}"
        result = split_by_structure(text, chunk_size=300, overlap=50)
        assert len(result) > 2  # short + multiple from long

    def test_short_paragraph_not_split_with_chunk_size(self):
        from app.chunking import split_by_structure

        text = "short para\n\nanother short"
        result = split_by_structure(text, chunk_size=300, overlap=50)
        assert result == ["short para", "another short"]
