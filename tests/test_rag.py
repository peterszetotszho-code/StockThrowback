"""Tests for the RAG retrieval stack (pure parts + rank_bm25)."""

import numpy as np
import pytest

from src.rag.chunking import chunk_text
from src.rag.citation import extract_citations, support_score, verify_citations
from src.rag.embeddings import HashEmbedder
from src.rag.hybrid_retriever import reciprocal_rank_fusion
from src.rag.ids import make_chunk_id
from src.rag.reranker import HeuristicReranker
from src.rag.schemas import Chunk


def _chunk(chunk_id, source_type, text, date=None):
    return Chunk(chunk_id=chunk_id, source_type=source_type, source_id="s", text=text, date=date)


def test_make_chunk_id():
    assert make_chunk_id("news", "gdelt_0", 3) == "news:gdelt_0:00003"


def test_chunk_text_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert chunks
    assert all(len(c.split()) <= 30 for c in chunks)
    assert len(chunks) > 1


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c d", chunk_size=3, overlap=5)


def test_rrf_fusion():
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("b", 5.0), ("c", 4.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    scores = dict(fused)
    assert set(scores) == {"a", "b", "c"}
    # b is rank 2 in dense and rank 1 in sparse -> beats a (rank 1 dense only).
    assert scores["b"] > scores["a"]


def test_keyword_index_search():
    pytest.importorskip("rank_bm25")
    from src.rag.keyword_index import BM25KeywordIndex

    # A corpus must be large enough that terms have varied document
    # frequencies; with 2 docs, rank_bm25's IDF is zero for every term.
    idx = BM25KeywordIndex()
    idx.index(
        [
            _chunk("k:0:00000", "knowledge", "moving average crossover strategy"),
            _chunk("k:1:00000", "knowledge", "relative strength index overbought"),
            _chunk("k:2:00000", "knowledge", "bollinger bands volatility squeeze"),
            _chunk("k:3:00000", "knowledge", "macd histogram momentum divergence"),
            _chunk("k:4:00000", "knowledge", "volume weighted average price"),
        ]
    )
    results = idx.search("moving average crossover", top_k=3)
    assert results
    assert results[0][0] == "k:0:00000"


def test_keyword_index_empty_corpus():
    pytest.importorskip("rank_bm25")
    from src.rag.keyword_index import BM25KeywordIndex

    idx = BM25KeywordIndex()
    idx.index([])
    assert idx.search("anything") == []


def test_heuristic_reranker_components():
    assert HeuristicReranker.authority(_chunk("x", "knowledge", "t")) == 0.9
    assert HeuristicReranker.authority(_chunk("x", "news", "t")) == 0.6
    assert HeuristicReranker.keyword_overlap("moving average", "a moving average line") == 1.0
    assert HeuristicReranker.keyword_overlap("moving average", "rsi overbought") == 0.0


def test_extract_citations():
    assert extract_citations("see [1] and [2,3]") == [1, 2, 3]
    assert extract_citations("also [1][2] here") == [1, 2]
    assert extract_citations("no citations") == []


def test_gdelt_datetime_and_iso_date():
    from src.rag.indexers import _gdelt_datetime, _iso_date

    assert _gdelt_datetime("2024-03-01", "000000") == "20240301000000"
    assert _gdelt_datetime("2024-03-01", "235959") == "20240301235959"
    assert _iso_date("20240301T120000Z") == "2024-03-01"
    assert _iso_date("short") == ""


def test_build_news_chunks():
    from src.rag.indexers import build_news_chunks

    chunks = build_news_chunks(
        [{"title": "Big news", "url": "http://x", "date": "2024-03-01", "source_id": "g0"}]
    )
    assert len(chunks) == 1
    assert chunks[0].source_type == "news"
    assert chunks[0].date == "2024-03-01"
    assert chunks[0].text == "Big news"


def test_support_score_cosine():
    assert support_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert support_score(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_hash_embedder_deterministic():
    embedder = HashEmbedder(dim=32)
    a = embedder.embed("moving average")
    b = embedder.embed("moving average")
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-9


def test_get_embedder_falls_back_to_hash(monkeypatch):
    from src.rag import embeddings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _unavailable():
        raise ImportError("sentence-transformers not installed")

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", _unavailable)
    assert isinstance(embeddings.get_embedder(), embeddings.HashEmbedder)


def test_get_embedder_prefers_sentence_transformers(monkeypatch):
    from src.rag import embeddings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeEmbedder:
        pass

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", FakeEmbedder)
    assert isinstance(embeddings.get_embedder(), FakeEmbedder)


def test_openai_embedder_records_usage():
    from src.rag.embeddings import OpenAIEmbedder
    from src.observability.usage_tracker import get_usage_tracker, reset_usage_tracker

    class FakeUsage:
        prompt_tokens = 7

    class FakeEmbedding:
        embedding = [0.1, 0.2, 0.3]

    class FakeResponse:
        data = [FakeEmbedding()]
        usage = FakeUsage()

    class FakeEmbeddings:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        embeddings = FakeEmbeddings()

    reset_usage_tracker()
    embedder = OpenAIEmbedder(client=FakeClient())
    vector = embedder.embed("hello")
    assert list(vector) == [0.1, 0.2, 0.3]

    records = get_usage_tracker()._snapshot()
    assert records[0].operation == "embed"
    assert records[0].model == "text-embedding-3-small"
    assert records[0].input_tokens == 7


def test_verify_citations():
    embed = HashEmbedder(dim=64)
    retrieved = {
        "k:1:00000": _chunk("k:1:00000", "knowledge", "moving average crossover"),
        "k:2:00000": _chunk("k:2:00000", "knowledge", "rsi overbought signal"),
    }
    citation_map = {1: "k:1:00000", 2: "k:99999999"}  # 2 is an orphan.
    report = "The moving average crossover triggers a buy. [1] The RSI is overbought. [2]"
    rep = verify_citations(report, citation_map, retrieved, embed.embed, threshold=0.0)
    assert rep.total_citations == 2
    assert rep.orphan_citations == 1
    assert rep.cited_sentences == 2
    assert rep.total_sentences == 3
    assert rep.coverage == pytest.approx(2 / 3)
