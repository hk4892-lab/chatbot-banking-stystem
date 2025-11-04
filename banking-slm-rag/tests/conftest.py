from __future__ import annotations

import hashlib
import os
from typing import Iterable, List

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api
from app import deps
from app.config import get_settings
from app.pipeline import ChatPipeline
from app.retrieval import FaissRetriever


class FakeEmbeddingModel:
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def encode(self, texts: Iterable[str], convert_to_numpy: bool = True, normalize_embeddings: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        vectors: List[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().split():
                digest = hashlib.sha1(token.encode('utf-8')).hexdigest()
                idx = int(digest[:4], 16) % self.dimension
                vec[idx] += 1.0
            if not np.any(vec):
                vec[0] = 1.0
            vectors.append(vec)
        array = np.vstack(vectors)
        if normalize_embeddings or convert_to_numpy:
            norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
            array = array / norms
        return array


class FakeGenerator:
    def __init__(self) -> None:
        self.last_passages = []

    def make_prompt(self, query, passages, lang):
        self.last_passages = list(passages)
        context_lines = [f"[{chunk.title} ({chunk.doc_id})]\n{chunk.content}" for chunk in self.last_passages]
        prompt = f"User ({lang}): {query}\nCONTEXT:\n" + '\n\n'.join(context_lines)
        return prompt, context_lines

    def generate(self, prompt: str, **_: object) -> str:
        if not self.last_passages:
            return 'I’m sorry, I cannot help with that. Please use official banking channels.'
        parts = []
        for chunk in self.last_passages:
            parts.append(f"{chunk.content.split('.')[0].strip()} [{chunk.title} ({chunk.doc_id})]")
        return ' '.join(parts)


@pytest.fixture(scope='session', autouse=True)
def configure_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    temp_dir = tmp_path_factory.mktemp('indices')
    os.environ['FAISS_INDEX_PATH'] = str(temp_dir / 'kb.faiss')
    os.environ['FAISS_META_PATH'] = str(temp_dir / 'kb.meta.pkl')
    os.environ['AUDIT_LOG_PATH'] = str(temp_dir / 'audit.log')
    os.environ['DEVICE_PREFERENCE'] = 'cpu'
    deps.get_pipeline.cache_clear()
    deps.get_generator.cache_clear()
    deps.get_retriever.cache_clear()
    deps.get_embedding_model.cache_clear()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(scope='session')
def fake_embeddings() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture(scope='session')
def retriever(fake_embeddings: FakeEmbeddingModel) -> FaissRetriever:
    settings = get_settings()
    retriever = FaissRetriever(settings=settings, embedding_model=fake_embeddings, auto_load=False)
    retriever._build_from_source()
    return retriever


@pytest.fixture(scope='session')
def pipeline(retriever: FaissRetriever) -> ChatPipeline:
    fake_generator = FakeGenerator()
    return ChatPipeline(settings=get_settings(), retriever=retriever, generator=fake_generator)


@pytest.fixture(scope='session')
def api_client(pipeline: ChatPipeline) -> TestClient:
    api.app.dependency_overrides[deps.get_pipeline] = lambda: pipeline
    client = TestClient(api.app)
    yield client
    api.app.dependency_overrides.clear()
