import json

import pytest
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from config import DEFAULT_CORPUS, get_chroma_path
from get_embedding_function import get_embedding_function
from query_data import LLM_MODEL, query_rag

CASES = [
    ('What is the global prevelance of MASLD?','Papers report conflicting statistics of 38·2% and 32.4%. The latter is cited from a more recent source, 2023 compared to 2019.'),
    ('What does MASH stand for?','Metabolic dysfunction-associated steatohepatitis')
]

@pytest.fixture(scope="session")
def db():
    path = get_chroma_path(DEFAULT_CORPUS)
    assert path.is_dir(), f"No store at {path}. Run populate_database.py first."
    return Chroma(persist_directory=str(path), embedding_function=get_embedding_function())


@pytest.fixture(scope="session")
def llm():
    return ChatOllama(
        model=LLM_MODEL, reasoning=False, num_ctx=8192, num_predict=512, temperature=0.2
    )

@pytest.mark.parametrize("question,expected", CASES)
def test_rag(question, expected,db,llm):
    actual = query_rag(question,db,llm)
    with open("results.jsonl", "a") as f:
        f.write(json.dumps({"q": question, "expected": expected, "actual": actual}) + "\n")
    assert expected in actual, f"expected {expected!r}, got {actual!r}"