# Notes (personal)

## Decisions
- **Chunking:** Docling `HybridChunker`, 512 tokens, tokenizer `all-MiniLM-L6-v2`. Docling's own default is 256 (gave ~326 chunks vs 180 at 512).
  - TODO: switch `TOKENIZER_ID` to `BAAI/bge-m3` to match the embedding model (needs `--reset`).
- **Embeddings:** `bge-m3` via Ollama. **LLM:** `qwen3:4b-instruct`. (`qwen3:4b` is the thinking-only 2507 release; it never finished answers.)
- **Chunk IDs:** `filename:page:index` (first page of the chunk). Chroma metadata is flattened to `source` (filename), `page`, `headings`, `id`.
- **Stores:** `data/<corpus>` -> `vector_stores/chroma_<corpus>`, chosen with `--corpus`.

## Rebuild with `--reset` when
- `MAX_TOKENS`, `TOKENIZER_ID` or the embedding model changes
- Docling version or pipeline changes
- a PDF is edited, replaced or renamed

## Ideas / TODO
- Label each chunk with its source in the prompt, and tell the model to report conflicting figures (see `evaluation/failure_modes.md` #1).
- Small-to-big retrieval: embed small chunks, return neighbouring chunks or the whole section.
- Score threshold on retrieval; reranking; hybrid (keyword + vector) search.
- Skip re-parsing PDFs already in the store (parsing is the slow part of every populate run).
- Hodgkin-Huxley corpus: the 1952 paper is probably scanned, so check OCR quality (equations).
