"""
Abstract base class for retrieval — now actually used by retrieve.py.
"""
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, question: str, company: str = None, year: str = None, top_k: int = 8) -> list:
        """Return list of dicts with keys: score, company, year, page, chunk, source."""
        ...
