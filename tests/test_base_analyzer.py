from unittest.mock import MagicMock, patch

from nlp_predictor.base_analyzer import BaseAnalyzer
from nlp_predictor.config import config
from nlp_predictor.lower_path.lower_path_analyzer import LowerPathAnalyzer


class DummyAnalyzer(BaseAnalyzer):
    def process_data(self, *args, **kwargs):
        pass


def test_base_analyzer_config_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    analyzer = DummyAnalyzer()
    assert analyzer.max_workers == config.hyperparams.models.max_workers
    assert analyzer.request_delay == config.hyperparams.models.request_delay


def test_base_analyzer_custom_params(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    analyzer = DummyAnalyzer(max_workers=5, request_delay=0.05)
    assert analyzer.max_workers == 5
    assert analyzer.request_delay == 0.05


def test_parallel_map_preserves_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    analyzer = DummyAnalyzer(max_workers=2, request_delay=0.0)
    items = list(range(10))
    results = analyzer._parallel_map(lambda x: x * 2, items)
    assert results == [x * 2 for x in items]


def test_parallel_map_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    analyzer = DummyAnalyzer()
    assert analyzer._parallel_map(lambda x: x, []) == []


def test_embeddings_batch_chunking(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    with patch("nlp_predictor.base_analyzer.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def fake_create(input, model):
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1, 0.2]) for _ in input]
            return mock_resp

        mock_client.embeddings.create.side_effect = fake_create

        analyzer = LowerPathAnalyzer.__new__(LowerPathAnalyzer)
        analyzer.client = mock_client

        # 250 items should produce 3 chunks: 100, 100, 50
        texts = [f"text_{i}" for i in range(250)]
        embeddings = analyzer._generate_embeddings_batch(texts)

        assert len(embeddings) == 250
        assert mock_client.embeddings.create.call_count == 3
        calls = mock_client.embeddings.create.call_args_list
        assert len(calls[0].kwargs["input"]) == 100
        assert len(calls[1].kwargs["input"]) == 100
        assert len(calls[2].kwargs["input"]) == 50
