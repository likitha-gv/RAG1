# RAG over PDFs

Ask questions about a set of PDFs using local models. PDFs are parsed and chunked with Docling, embedded with `bge-m3` into Chroma, and answered by a local LLM through Ollama.

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull bge-m3
   ollama pull qwen3:4b-instruct
   ```
2. Install the Python packages:
   ```
   pip install docling langchain-docling langchain-chroma langchain-ollama langchain-core transformers
   ```
3. Put your PDFs in the `data/` folder (not tracked by git).

## Run

Build the database (first run is slow while Docling parses the PDFs):

```
python populate_database.py
```

Ask questions (type `quit` to exit):

```
python query_data.py
```

Re-running `populate_database.py` only adds new chunks. If you change the chunk size, tokenizer, embedding model or PDFs, rebuild from scratch:

```
python populate_database.py --reset
```

## Options for `query_data.py`

Flags go on the start command and apply to the whole session.

| Flag | Effect |
|---|---|
| `--show-chunks` | Print each retrieved chunk with its score, page, section and text |
| `--show-prompt` | Print the full prompt sent to the LLM |
| `-k N` | Retrieve N chunks (default 5) |

```
python query_data.py --show-chunks -k 3
```

## Configuration

Set at the top of each script:

- `populate_database.py`: `DATA_PATH`, `CHROMA_PATH`, `TOKENIZER_ID`, `MAX_TOKENS` (chunk size)
- `query_data.py`: `CHROMA_PATH`, `K`, `LLM_MODEL`, `PROMPT_TEMPLATE`
- `get_embedding_function.py`: embedding model

`CHROMA_PATH` must match in both scripts.

## Notebooks

`populate_db_test.ipynb` and `query_test.ipynb` are the step-by-step versions used for testing.
