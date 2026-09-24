"""Citation verification: existence, support score, and coverage.

This is a heuristic signal, not a truth guarantee — the report states so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .schemas import Chunk

_CIT_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")
_CIT_MARKER_RE = re.compile(r"\[\d+(?:[,，]\s*\d+)*\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationResult:
    citation_number: int
    chunk_id: str
    exists: bool
    support_score: float
    supported: bool
    claim_sentence: str


@dataclass
class CitationReport:
    results: list[CitationResult] = field(default_factory=list)
    threshold: float = 0.5
    total_sentences: int = 0
    cited_sentences: int = 0
    total_citations: int = 0
    orphan_citations: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of report sentences carrying >= 1 citation (0.0..1.0)."""
        return (self.cited_sentences / self.total_sentences) if self.total_sentences else 0.0

    @property
    def orphan_rate(self) -> float:
        """Fraction of citation numbers that resolve to no retrieved chunk."""
        return (self.orphan_citations / self.total_citations) if self.total_citations else 0.0


def extract_citations(text: str) -> list[int]:
    """Parse inline markers like ``[1]``, ``[1][2]``, ``[1,2]`` into sorted ints."""
    numbers: set[int] = set()
    for match in _CIT_RE.finditer(text):
        for part in re.split(r"[,，]", match.group(1)):
            part = part.strip()
            if part.isdigit():
                numbers.add(int(part))
    return sorted(numbers)


def support_score(claim_embedding: np.ndarray, chunk_embedding: np.ndarray) -> float:
    """Cosine similarity between a claim sentence and its cited chunk."""
    a = np.asarray(claim_embedding, dtype=float)
    b = np.asarray(chunk_embedding, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def verify_citations(
    report_text: str,
    citation_map: Mapping[int, str],
    retrieved: Mapping[str, Chunk],
    embed_fn: Callable[[str], np.ndarray],
    threshold: float = 0.5,
) -> CitationReport:
    """Verify citations: existence, support, and coverage.

    Splits ``report_text`` into sentences, finds the sentence holding each
    ``[n]``, embeds it, and compares to the cited chunk. Flags missing
    citations and low-support claims without fabricating verdicts.

    Args:
        report_text: The generated Markdown report.
        citation_map: ``citation_number -> chunk_id`` mapping.
        retrieved: ``chunk_id -> Chunk`` map of the retrieved set.
        embed_fn: Reuses the dense index embedder for comparable distances.
        threshold: Cosine similarity above which a claim is considered supported.

    Returns:
        A ``CitationReport`` with per-citation results and coverage metrics.
    """
    sentences = [s.strip() for s in _SENT_RE.split(report_text) if s.strip()]
    results: list[CitationResult] = []
    cited_indexes: set[int] = set()

    for index, sentence in enumerate(sentences):
        numbers = extract_citations(sentence)
        if numbers:
            cited_indexes.add(index)
        for number in numbers:
            chunk_id = citation_map.get(number)
            exists = chunk_id is not None and chunk_id in retrieved
            if not exists:
                results.append(CitationResult(number, chunk_id or "", False, 0.0, False, sentence))
                continue
            chunk = retrieved[chunk_id]
            claim = _CIT_MARKER_RE.sub("", sentence).strip()
            score = support_score(embed_fn(claim), embed_fn(chunk.text))
            results.append(CitationResult(number, chunk_id, True, score, score >= threshold, sentence))

    return CitationReport(
        results=results,
        threshold=threshold,
        total_sentences=len(sentences),
        cited_sentences=len(cited_indexes),
        total_citations=len(results),
        orphan_citations=sum(1 for r in results if not r.exists),
    )
