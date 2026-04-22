from abc import ABC, abstractmethod

from src.data.schemas import ChunkRecord


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, chunks: list[ChunkRecord], prompt: str) -> str:
        raise NotImplementedError
