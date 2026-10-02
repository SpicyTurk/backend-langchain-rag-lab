"""Response formatting helpers for the LangChain RAG API."""

CHAIN_EXPRESSION = "ChatPromptTemplate | ChatOllama | StrOutputParser"

COMPONENTS = {
    "vector_store": "Chroma",
    "retrieval_method": "similarity_search_with_score",
    "prompt_template": "ChatPromptTemplate",
    "chat_model": "ChatOllama",
    "output_parser": "StrOutputParser",
}

SCORE_TYPE = "Chroma distance; lower usually means closer in this lesson setup"

FALLBACK_ANSWER = (
    "I do not have enough approved runbook context to answer that reliably."
)

def _metadata_value(metadata, key, default):
    """Return a metadata value, using a default when it is missing or empty."""
    value = (metadata or {}).get(key)
    return value if value else default

def format_sources(scored_documents):
    """Format retrieved documents as source metadata for the API response."""

    sources = []
    for document, distance in scored_documents:
        metadata = getattr(document, "metadata", None) or {}
        sources.append(
            {
                "source_id": _metadata_value(metadata, "source_id", "unknown"),
                "title": _metadata_value(metadata, "title", "Untitled"),
                "category": _metadata_value(metadata, "category", "Uncategorized"),
                "section": _metadata_value(metadata, "section", "Unspecified"),
                "chunk_id": _metadata_value(metadata, "chunk_id", "unknown"),
                "distance": round(float(distance), 4),
            }
        )
    return sources


def format_langchain_debug(scored_documents, context, top_k, fallback):
    """Return LangChain debug metadata for inspectability."""

    chunk_ids = [
        _metadata_value(getattr(document, "metadata", None), "chunk_id", "unknown")
        for document, _distance in scored_documents
    ]

    return {
        "chain_expression": CHAIN_EXPRESSION,
        "components": COMPONENTS,
        "retrieved_count": len(scored_documents),
        "retrieved_chunk_ids": chunk_ids,
        "top_k": top_k,
        "context_characters": len(context),
        "fallback": fallback,
        "score_type": SCORE_TYPE,
    }


def format_success_response(answer, sources, debug):
    """Format a successful RAG response."""

    return {
        "answer": answer.strip(),
        "sources": sources,
        "langchain": debug,
    }


def format_fallback_response(debug):
    """Format a safe response when no usable context is available."""

    return {
        "answer": FALLBACK_ANSWER,
        "sources": [],
        "langchain": debug,
    }


def format_error_response(error, message):
    """Format an API error response."""

    return {"error": error, "message": message}