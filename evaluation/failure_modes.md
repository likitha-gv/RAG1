# Failure modes

Observed problems with the pipeline, what caused them, and what was done. Add new entries as they come up.

Entry format: **Query** (what was asked) / **Observed** / **Cause** / **Status**.

## 1. Conflicting statistics, answer depends on wording

- **Query:** "What is the current global prevalence of MASLD?" vs "How many adults globally have MASLD?"
- **Observed:** the first answers 38.2% (Li 2026), the second 32.4% (Lazarus 2025). Neither answer mentions the other figure.
- **Cause:** both chunks are retrieved; the model picks the one whose wording is closest to the question. The prompt has no source names or years, so it cannot compare papers, and the template does not say what to do when sources disagree.
- **Status:** open. Planned: label each chunk with its source in the prompt, and instruct the model to report each figure with its source when they differ.

## 2. Thinking model never gives a final answer

- **Query:** "What cells drive liver fibrosis in MASH?"
- **Observed:** with `qwen3:4b`, the output was reasoning only and stopped mid-sentence at the 512-token `num_predict` cap.
- **Cause:** `qwen3:4b` is the thinking-only 2507 release, so `reasoning=False` does not turn thinking off.
- **Status:** fixed by switching to `qwen3:4b-instruct`.

## 3. Stale or mixed database after changing settings

- **Observed:** not yet seen; known risk.
- **Cause:** chunk IDs are `source:page:index`, so changing `MAX_TOKENS`, the tokenizer, or the embedding model leaves old chunks in place and the "already exists" check skips new ones.
- **Status:** mitigated. Rebuild with `python src/populate_database.py --reset` after any such change.

## 4. Multi-line pasted question is split

- **Observed:** a question pasted with a line break was read as two questions; the second became a garbled query.
- **Cause:** `input()` reads one line at a time.
- **Status:** open (minor). Enter each question on a single line.
