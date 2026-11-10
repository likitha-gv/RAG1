import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"  # must match the populate script
K = 5  # chunks to retrieve
LLM_MODEL = "qwen3:4b-instruct"  # an Ollama model you have pulled

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# optional detailed chunk inspection (to test retrieval performance)
def show_results(results):
    for rank, (doc, score) in enumerate(results, start=1):
        m = doc.metadata
        print(f"[{rank}] score={score:.3f}  page={m.get('page')}  file={m.get('source', '').split('/')[-1]}")
        print(f"    section: {m.get('headings')}")
        print(f"    id: {m.get('id')}")
        print(f"    text: {doc.page_content}\n")


def query_rag(query_text: str, db, llm, k: int = K, show_prompt: bool = False, show_chunks: bool = False):
    # return closest chunks
    results =db.similarity_search_with_score(query_text, k=k)
    if show_chunks:
        show_results(results)

    context_text = context_text = "\n\n---\n\n".join(
    f"[Source: {doc.metadata.get('source', '').split('/')[-1]}, p.{doc.metadata.get('page')}]\n{doc.page_content}"
    for doc, _score in results
)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )
    if show_prompt:
        print(prompt)

    # Ask LLM 
    response_text = llm.invoke(prompt).content

    # Answer and sources only 
    sources = [
        f"{doc.metadata.get('source', '').split('/')[-1]} (p.{doc.metadata.get('page')}) "
        f"- {doc.metadata.get('headings')}"
        for doc, _score in results
    ]
    print(f"\nResponse: {response_text}\n")
    print("Sources:")
    for s in sources:
        print(f"  - {s}")
    return response_text


def main():
    # Create CLI.
    parser = argparse.ArgumentParser(description="Ask questions about your documents.")
    parser.add_argument(
        "query_text",
        type=str,
        nargs="?",
        help="The question. If omitted, starts an interactive session.",
    )
    parser.add_argument("-k", type=int, default=K, help=f"Chunks to retrieve (default {K}).")
    parser.add_argument("--show-prompt", action="store_true", help="Print the full prompt sent to the LLM.")
    parser.add_argument("--show-chunks", action="store_true", help="Print each retrieved chunk with its score.")
    args = parser.parse_args()

    # Build the database connection and the LLM ONCE; every question reuses them.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    llm = ChatOllama(
        model=LLM_MODEL,
        reasoning=False,   # no effect on instruct models; stops thinking on hybrid ones
        num_ctx=8192,      # context window: room for k chunks + the answer
        num_predict=512,   # hard cap on answer length, so it cannot run forever
        temperature=0.2,   # low = sticks closer to the context
    )
    print(f"Chunks in database: {len(db.get(include=[])['ids'])}")

    options = dict(k=args.k, show_prompt=args.show_prompt, show_chunks=args.show_chunks)

    # Otherwise: interactive session, reusing db and llm for every question.
    while True:
        try:
            q = input("\nQuestion (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C
            print()
            break
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        query_rag(q, db, llm, **options)


if __name__ == "__main__":
    main()
