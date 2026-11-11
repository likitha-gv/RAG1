# RAG over PDFs

Ask questions about a set of PDFs using local models. PDFs are parsed and chunked with Docling, embedded with `bge-m3` into Chroma, and answered by a local LLM through Ollama.

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull bge-m3
   ollama pull qwen3:4b-instruct
   ```
   Ollama must be running (the desktop app, or `ollama serve`) whenever you build or query.
2. Install the Python packages:
   ```
   pip install docling langchain-docling langchain-chroma langchain-ollama langchain-core transformers
   ```
3. Put your PDFs in a folder under `data/`, one folder per document set, e.g. `data/my_corpus/` (not tracked by git).

## Run

Run commands from the project root. Each folder under `data/` is a *corpus* with its own store in `vector_stores/chroma_<corpus>/`. Commands without `--corpus` use the default corpus, set by `DEFAULT_CORPUS` in `src/config.py` (`mash` as shipped). Add `--corpus NAME` to either script to use another folder under `data/`, and use the same name for both scripts.

Build the database (first run is slow while Docling parses the PDFs):

```
python src/populate_database.py                        # default corpus
python src/populate_database.py --corpus my_corpus     # data/my_corpus
```

Ask questions (type `quit` to exit):

```
python src/query_data.py                               # default corpus
python src/query_data.py --corpus my_corpus
```

Re-running `populate_database.py` only adds new chunks. If you change the chunk size, tokenizer, embedding model or PDFs, rebuild from scratch:

```
python src/populate_database.py --reset                        # default corpus
python src/populate_database.py --corpus my_corpus --reset
```

`--reset` only clears the selected corpus's store.

## Options for `query_data.py`

Flags go on the start command and apply to the whole session.

| Flag | Effect |
|---|---|
| `--show-chunks` | Print each retrieved chunk with its score, page, section and text |
| `--show-prompt` | Print the full prompt sent to the LLM |
| `-k N` | Retrieve N chunks (default 5) |
| `--corpus NAME` | Which store to query (default: `DEFAULT_CORPUS`) |

```
python src/query_data.py --show-chunks -k 3
```

## Configuration

Set at the top of each script:

- `src/populate_database.py`: `TOKENIZER_ID`, `MAX_TOKENS` (chunk size)
- `src/query_data.py`: `K`, `LLM_MODEL`, `PROMPT_TEMPLATE`
- `src/get_embedding_function.py`: embedding model
- `src/config.py`: default corpus and folder locations

## Notebooks

`notebooks/populate_db_test.ipynb` and `notebooks/query_test.ipynb` are the step-by-step versions used for testing. Choose the document set with the `CORPUS` variable in each notebook's settings cell (it defaults to `DEFAULT_CORPUS`); it must be the same in both.
