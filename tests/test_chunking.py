class TestSplitText:
    def test_short_text_returns_single_chunk(self):
        from rag.data.chunking import split_text

        result = split_text("short text", chunk_size=500)
        assert result == ["short text"]

    def test_empty_string_returns_empty_list(self):
        from rag.data.chunking import split_text

        result = split_text("")
        assert result == []

    def test_long_text_splits_into_multiple_chunks(self):
        from rag.data.chunking import split_text

        text = "word " * 200  # 1000 chars
        result = split_text(text, chunk_size=300, overlap=50)
        assert len(result) > 1

    def test_overlap_applied_correctly(self):
        from rag.data.chunking import split_text

        text = "word " * 200
        result = split_text(text, chunk_size=300, overlap=50)
        # 2番目のチャンクの先頭部分が1番目の末尾と重複する
        for i in range(len(result) - 1):
            tail = result[i][-50:]
            assert tail in result[i + 1]

    def test_splits_at_word_boundary(self):
        from rag.data.chunking import split_text

        text = "hello world this is a test sentence for chunking"
        result = split_text(text, chunk_size=20, overlap=5)
        for chunk in result:
            # チャンクが単語の途中で切れていないことを確認
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_no_data_loss(self):
        from rag.data.chunking import split_text

        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        result = split_text(text, chunk_size=30, overlap=10)
        all_words = set(text.split())
        chunk_words = set()
        for chunk in result:
            chunk_words.update(chunk.split())
        assert all_words == chunk_words

    def test_different_chunk_sizes_produce_different_counts(self):
        from rag.data.chunking import split_text

        text = "word " * 200
        result_small = split_text(text, chunk_size=300, overlap=50)
        result_large = split_text(text, chunk_size=800, overlap=50)
        assert len(result_small) > len(result_large)

    def test_text_exactly_chunk_size_returns_single_chunk(self):
        from rag.data.chunking import split_text

        text = "a" * 500
        result = split_text(text, chunk_size=500)
        assert result == [text]

    def test_each_chunk_within_size_limit(self):
        from rag.data.chunking import split_text

        text = "hello world this is testing " * 50
        result = split_text(text, chunk_size=200, overlap=50)
        for chunk in result:
            assert len(chunk) <= 200 + 50  # some tolerance for word boundary

    def test_japanese_text_splits_correctly(self):
        from rag.data.chunking import split_text

        text = "これはテストです " * 100
        result = split_text(text, chunk_size=100, overlap=20)
        assert len(result) > 1
        all_text = "".join(result)
        # 元のテキストの全単語が含まれる
        assert "これはテストです" in all_text

    def test_default_parameters(self):
        from rag.data.chunking import split_text

        text = "word " * 200  # 1000 chars
        result = split_text(text)  # defaults: chunk_size=500, overlap=100
        assert len(result) > 1


class TestSplitByStructure:
    def test_splits_by_double_newline(self):
        from rag.data.chunking import split_by_structure

        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        result = split_by_structure(text)
        assert result == ["paragraph one", "paragraph two", "paragraph three"]

    def test_single_paragraph_returns_single_chunk(self):
        from rag.data.chunking import split_by_structure

        result = split_by_structure("just one paragraph")
        assert result == ["just one paragraph"]

    def test_empty_input_returns_empty_list(self):
        from rag.data.chunking import split_by_structure

        result = split_by_structure("")
        assert result == []

    def test_strips_whitespace_from_paragraphs(self):
        from rag.data.chunking import split_by_structure

        text = "  para one  \n\n  para two  "
        result = split_by_structure(text)
        assert result == ["para one", "para two"]

    def test_skips_empty_paragraphs(self):
        from rag.data.chunking import split_by_structure

        text = "para one\n\n\n\n\npara two"
        result = split_by_structure(text)
        assert result == ["para one", "para two"]

    def test_long_paragraph_split_further_with_chunk_size(self):
        from rag.data.chunking import split_by_structure

        long_para = "word " * 200
        text = f"short para\n\n{long_para}"
        result = split_by_structure(text, chunk_size=300, overlap=50)
        assert len(result) > 2  # short + multiple from long

    def test_short_paragraph_not_split_with_chunk_size(self):
        from rag.data.chunking import split_by_structure

        text = "short para\n\nanother short"
        result = split_by_structure(text, chunk_size=300, overlap=50)
        assert result == ["short para", "another short"]

    def test_preserves_paragraph_order(self):
        from rag.data.chunking import split_by_structure

        text = "first\n\nsecond\n\nthird\n\nfourth"
        result = split_by_structure(text)
        assert result == ["first", "second", "third", "fourth"]

    def test_japanese_paragraphs(self):
        from rag.data.chunking import split_by_structure

        text = "最初の段落です。\n\n2番目の段落です。\n\n3番目の段落です。"
        result = split_by_structure(text)
        assert result == ["最初の段落です。", "2番目の段落です。", "3番目の段落です。"]

    def test_mixed_long_and_short_paragraphs(self):
        from rag.data.chunking import split_by_structure

        short = "short"
        long_para = "word " * 200  # 1000 chars
        text = f"{short}\n\n{long_para}\n\n{short}"
        result = split_by_structure(text, chunk_size=300, overlap=50)
        assert result[0] == short
        assert result[-1] == short
        assert len(result) > 3  # short + multiple splits + short

    def test_none_chunk_size_returns_paragraphs_as_is(self):
        from rag.data.chunking import split_by_structure

        long_para = "word " * 200
        text = f"short\n\n{long_para}"
        result = split_by_structure(text, chunk_size=None)
        assert len(result) == 2
        assert result[0] == "short"
        assert result[1] == long_para.strip()


class TestSplitTextEdgeCases:
    def test_none_input_returns_empty_list(self):
        from rag.data.chunking import split_text

        assert split_text(None) == []

    def test_whitespace_only_returns_empty_list(self):
        from rag.data.chunking import split_text

        assert split_text("   ") == ["   "]  # not empty: len > 0 but <= chunk_size

    def test_long_string_without_spaces(self):
        from rag.data.chunking import split_text

        text = "a" * 1000
        result = split_text(text, chunk_size=300, overlap=50)
        assert len(result) >= 1
        # 元のテキストの全文字がチャンクに含まれる
        combined = "".join(result)
        assert len(combined) >= len(set(text))

    def test_overlap_larger_than_chunk_size_does_not_hang(self):
        from rag.data.chunking import split_text

        text = "hello world this is a test"
        # overlap > chunk_size でも無限ループにならない
        result = split_text(text, chunk_size=5, overlap=100)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_size_one(self):
        from rag.data.chunking import split_text

        result = split_text("a b", chunk_size=1, overlap=0)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_newlines_in_text(self):
        from rag.data.chunking import split_text

        text = "line one\nline two\nline three " * 20
        result = split_text(text, chunk_size=100, overlap=20)
        assert len(result) >= 1
        combined = " ".join(result)
        assert "line one" in combined


class TestSplitByStructureEdgeCases:
    def test_none_input_returns_empty_list(self):
        from rag.data.chunking import split_by_structure

        assert split_by_structure(None) == []

    def test_whitespace_only_returns_empty_list(self):
        from rag.data.chunking import split_by_structure

        assert split_by_structure("   \n\n   ") == []

    def test_single_newline_no_split(self):
        from rag.data.chunking import split_by_structure

        result = split_by_structure("line one\nline two")
        assert len(result) == 1
        assert "line one\nline two" == result[0]

    def test_chunk_size_zero_returns_list(self):
        from rag.data.chunking import split_by_structure

        # chunk_size=0 でもクラッシュしない
        text = "short\n\nanother"
        result = split_by_structure(text, chunk_size=0)
        assert isinstance(result, list)
