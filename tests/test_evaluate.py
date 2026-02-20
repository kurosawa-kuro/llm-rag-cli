import json
import os
import re
from unittest.mock import patch, MagicMock, mock_open
import pytest


class TestEvalQuestions:
    def test_eval_questions_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        assert os.path.exists(path)

    def test_eval_questions_is_valid_json_list(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_eval_questions_has_minimum_count(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 10

    def test_each_question_has_required_fields(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        for i, q in enumerate(data):
            assert "query" in q, f"Question {i} missing 'query'"
            assert "expected_source" in q, f"Question {i} missing 'expected_source'"
            assert "expected_keywords" in q, f"Question {i} missing 'expected_keywords'"

    def test_expected_source_matches_naming_convention(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        pattern = re.compile(r"^.+\.(csv|pdf):(r|p)\d+$")
        for i, q in enumerate(data):
            assert pattern.match(q["expected_source"]), \
                f"Question {i} source '{q['expected_source']}' doesn't match pattern"

    def test_expected_keywords_is_nonempty_list(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        for i, q in enumerate(data):
            assert isinstance(q["expected_keywords"], list)
            assert len(q["expected_keywords"]) >= 1
