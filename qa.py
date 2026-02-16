import os
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

# =========================================================
# ENV
# =========================================================
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore").strip()
TOP_K = int(os.getenv("DOC_TOP_K", "4"))
DOC_MAX_DISTANCE = float(os.getenv("DOC_MAX_DISTANCE", "0.85"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

_store: Optional[FAISS] = None


def _get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        api_key=OPENAI_API_KEY,
    )


def _load_store() -> FAISS:
    global _store
    if _store is None:
        _store = FAISS.load_local(
            VECTORSTORE_DIR,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _store


def reload_store():
    global _store
    _store = None


def ask_docs(query: str) -> Optional[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return None

    store = _load_store()
    results: List[Tuple[Any, float]] = store.similarity_search_with_score(query, k=TOP_K)

    if not results:
        return None

    best_doc, best_score = results[0]
    if float(best_score) > DOC_MAX_DISTANCE:
        return None

    context = "\n\n---\n\n".join(d.page_content for d, _ in results)
    sources = list({
        f"{d.metadata.get('source')} (page {d.metadata.get('page')})"
        for d, _ in results
    })

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "Answer ONLY using the provided SOP context. "
        "If answer is not present, reply exactly: NOT_FOUND."
    )

    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"},
        ],
        temperature=0,
    )

    answer = resp.choices[0].message.content.strip()
    if answer == "NOT_FOUND":
        return None

    return {"answer": answer, "sources": sources}


def ask_openai(query: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()
