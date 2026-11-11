from pathlib import Path

# Paths are anchored to the project root, so the scripts work from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
STORES_ROOT = PROJECT_ROOT / "vector_stores"

DEFAULT_CORPUS = "mash"  # folder name under data/; override with --corpus


def get_data_path(corpus: str) -> Path:
    # e.g. corpus "mash" -> <project>/data/mash
    return DATA_ROOT / corpus


def get_chroma_path(corpus: str) -> Path:
    # e.g. corpus "mash" -> <project>/vector_stores/chroma_mash
    return STORES_ROOT / f"chroma_{corpus}"


def available_corpora() -> list[str]:
    # Folders under data/ that could be built into a store.
    if not DATA_ROOT.is_dir():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


def available_stores() -> list[str]:
    # Corpus names that already have a built store.
    if not STORES_ROOT.is_dir():
        return []
    return sorted(
        p.name.removeprefix("chroma_")
        for p in STORES_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("chroma_")
    )
