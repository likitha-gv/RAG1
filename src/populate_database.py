import argparse
import shutil
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from transformers import AutoTokenizer

from config import DEFAULT_CORPUS, available_corpora, get_chroma_path, get_data_path
from get_embedding_function import get_embedding_function

# Tokenizer + max chunk size used by the chunker. Change these to match your embedding model.
# Larger MAX_TOKENS = bigger chunks with more surrounding context; smaller = more precise but fragmented.
# If you change either of these (or the embedding model), rebuild the database with --reset.
TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"Folder name under data/ to build a store from (default: {DEFAULT_CORPUS}).",
    ) 
   # set current corpus, defaults to DEFAULT_CORPUS in config.py, any other corpus should match the folder name within data 
   # i.e. data/hh/... enter --corpus hh 
   
    parser.add_argument("--reset", action="store_true", help="Reset this corpus's database first.")
    args = parser.parse_args()

    # Both paths come from the corpus name: data/<corpus> and vector_stores/chroma_<corpus>.
    data_path = get_data_path(args.corpus)
    chroma_path = get_chroma_path(args.corpus)
    if not data_path.is_dir():
        hint = " (the default corpus; pass --corpus NAME to choose another)" if args.corpus == DEFAULT_CORPUS else ""
        raise SystemExit(
            f"No folder {data_path}{hint}.\n"
            f"Corpora found under data/: {available_corpora() or 'none'}"
        ) 
    # if missing corpus is default, hint clarifies why the DEFAULT_CORPUS name was shown in the folder path (e.g. mash)
    # if someone types a wrong name, message skips default corpus hint, uses what their data folder path should be
    print(f"Corpus: {args.corpus}  (PDFs: {data_path}, store: {chroma_path})")

    # Check if the database should be cleared (using the --reset flag).
    if args.reset:
        print("✨ Clearing Database")
        clear_database(chroma_path)

    # Create (or update) the data store.
    # Docling chunks while it loads, so there is no separate split step.
    chunks = load_documents(data_path)
    add_to_chroma(chunks, chroma_path)


def load_documents(data_path):
    # Reads every PDF in data_path and returns Docling chunks (one Document per chunk).
    pdf_paths = [str(p) for p in data_path.glob("*.pdf")]
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {data_path}")
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_ID),
            max_tokens=MAX_TOKENS,
        )
    )
    loader = DoclingLoader(file_path=pdf_paths, chunker=chunker)
    chunks = loader.load()
    print(f"Loaded: {len(chunks)} chunks from {len(pdf_paths)} PDFs")
    return chunks


def get_pages(chunk):
    # A chunk can span pages; return all page numbers from its Docling doc_items.
    items = chunk.metadata["dl_meta"]["doc_items"]
    return sorted({p["page_no"] for item in items for p in item["prov"]})


def calculate_chunk_ids(chunks):
    # This will create IDs like "paper.pdf:3:2"
    # Source : Page Number (first page of the chunk) : Chunk Index on that page

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = Path(chunk.metadata.get("source")).name
        page = get_pages(chunk)[0]
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the chunk metadata.
        chunk.metadata["page"] = page
        chunk.metadata["id"] = chunk_id

    unique_ids = len({c.metadata["id"] for c in chunks})
    if unique_ids != len(chunks):
        print(f"⚠️  Only {unique_ids} unique IDs for {len(chunks)} chunks - some IDs collide!")

    return chunks


def flatten_metadata(chunk):
    # Chroma only accepts flat metadata (str/int/float/bool), so drop the nested dl_meta.
    # Returns a NEW Document so the original chunk keeps its full metadata.
    return Document(
        page_content=chunk.page_content,
        metadata={
            "source": Path(chunk.metadata["source"]).name,  # filename only, no machine path
            "page": chunk.metadata["page"],
            "headings": " > ".join(chunk.metadata["dl_meta"].get("headings", [])),
            "id": chunk.metadata["id"],
        },
    )


def add_to_chroma(chunks: list[Document], chroma_path):
    # Load the existing database.
    db = Chroma(
        persist_directory=str(chroma_path), embedding_function=get_embedding_function()
    )

    # Calculate Page IDs (also sets metadata["page"]).
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        # Flatten copies of the new chunks; the IDs were already computed above.
        new_chunks = [flatten_metadata(c) for c in new_chunks]
        new_chunk_ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)  # saves automatically
        print(f"DB now holds {len(db.get(include=[])['ids'])} chunks")
    else:
        print("✅ No new documents to add")


def clear_database(chroma_path):
    if chroma_path.exists():
        shutil.rmtree(chroma_path)


if __name__ == "__main__":
    main()
