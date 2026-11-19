# Changelog

Newest first.

## 2026-11-15
- Added `CLAUDE.md`, `src/hh/CLAUDE.md` and this changelog.

## 2026-11-11
- Restructured into `src/`, `notebooks/` and `evaluation/`.
- Added multi-corpus support: `--corpus` flag, `data/<corpus>/` and `vector_stores/chroma_<corpus>/` (paths in `src/config.py`).
- Chunk IDs and `source` metadata are now filename-only.
- Added clear errors for a missing corpus folder or store.
- Added `NOTES.md` and `evaluation/failure_modes.md`.

## Earlier
- Docling `HybridChunker` (512 tokens), Chroma store with `bge-m3` embeddings, interactive `query_data.py` using `qwen3:4b-instruct`.
