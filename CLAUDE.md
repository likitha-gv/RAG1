# Project Overview

## Final Goal

## Goal

The end goal is a RAG plus simulator tool for C. elegans neuron models, to be added to the research group's production RAG. It is built in stages, each working before the next starts:

1. **Simulator on its own** (current): simulate from a hand-written JSON input (`hh1952.json`) to test the simulator itself. No RAG.
2. **HH RAG + simulator:** retrieve parameters from HH papers and feed the simulator.
3. **C. elegans RAG + simulator.**

Current stage: 1. Update this line when the stage changes.

## Scope

- The RAG in this repo is a basic stand-in, built for learning and to build the tool on. **Do not optimise retrieval quality.** The production RAG replaces it, so keep the tool's interface to retrieval narrow (a query in, chunks with source and page out) so it can be swapped.
- Design for the later stages: don't hard-code things specific to HH in code that C. elegans models will need to share (for example, use a simulator dispatch, not one fixed model).
- The owner is learning, so briefly explain the reasoning behind changes.

## Layout

- `src/`: RAG scripts, run from the project root. `config.py` holds the corpus paths.
- `notebooks/`: step-by-step test versions of the scripts. They must stay consistent with `src/`.
- `evaluation/failure_modes.md`: observed failures and their causes.
- `NOTES.md`: the owner's personal notes. Don't edit unless asked.

## Project conventions

- **Corpora:** each folder in `data/<corpus>/` has its own store in `vector_stores/chroma_<corpus>/`. Both scripts take `--corpus`. Never hard-code a corpus name outside `config.py`.
- **Chunk IDs and `source` are filename-only** (`paper.pdf:3:2`), so no machine path enters the database.
- **Rebuild needed:** changing the chunk size, tokenizer, embedding model or PDFs makes the store stale, because ingestion only checks IDs. Say so and tell the user to run `--reset`.
- **Chroma metadata must be flat.** Docling's nested `dl_meta` is flattened before storage.
- **`query_data.py` is always an interactive session.** `db` and `llm` are built once. Flags go on the launch command, not inside the session.
- **Ollama LLM:** use a non-thinking model (`qwen3:4b-instruct`). Plain `qwen3:4b` is thinking-only and can exhaust `num_predict` before answering.
- The PDFs and `vector_stores/` are git-ignored. Don't commit them.
- Don't run the full populate script or a real query without asking. Both are slow and need Ollama.

# Working on the project

- Update `evaluation/failure_modes.md` when a new failure mode is found or one is resolved. Update `README.md` when a user-facing command, flag or setup step changes. Skip both for small edits.
- Keep a changelog in `CHANGELOG.md` (newest first, one line per notable change). Add an entry for each meaningful change, not every edit.
- Flag when `CLAUDE.md` itself is out of date, for example if the scope, workflow steps, conventions or layout change. Suggest the specific edit and wait for approval. Don't edit it silently.
- Don't commit or push unless asked.

# Behavioral guidelines

## Communication

Be concise and direct. Short sentences, no preamble or recap. Lead with the answer, then the reason. Use lists or tables when they are easier to scan than prose.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked. Suggestions in chat only if you think it's important/useful. 
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
