class TestBuildPrompt:
    def test_includes_japanese_instruction(self):
        from rag.components.prompting import build_prompt

        result = build_prompt("質問テスト", ["context1"])
        assert "簡潔に回答してください" in result

    def test_includes_query(self):
        from rag.components.prompting import build_prompt

        result = build_prompt("質問テスト", ["context1"])
        assert "質問:質問テスト" in result

    def test_includes_answer_marker(self):
        from rag.components.prompting import build_prompt

        result = build_prompt("質問", ["c1"])
        assert "回答:" in result

    def test_includes_contexts(self):
        from rag.components.prompting import build_prompt

        result = build_prompt("q", ["context content here"])
        assert "context content here" in result

    def test_empty_contexts(self):
        from rag.components.prompting import build_prompt

        result = build_prompt("q", [])
        assert "質問:q" in result

    def test_multiple_contexts(self):
        from rag.components.prompting import build_prompt

        result = build_prompt("q", ["c1", "c2", "c3"])
        assert "c1" in result
        assert "c2" in result
        assert "c3" in result
