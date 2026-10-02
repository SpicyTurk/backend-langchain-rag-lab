"""LangChain-supported RAG workflow service."""

from lib.config import CHAT_MODEL, DEFAULT_TOP_K
from lib.response_formatter import (
    format_fallback_response,
    format_langchain_debug,
    format_sources,
    format_success_response,
)
from lib.vector_store import retrieve_context


class LangChainServiceError(Exception):
    """Raised when the LangChain RAG service cannot complete a request."""


def build_chat_model():
    """Build the local chat model wrapper."""

    from langchain_ollama import ChatOllama

    return ChatOllama(model=CHAT_MODEL, temperature=0)


def build_chain():
    """Build the LangChain prompt to model to parser sequence."""

    from langchain_core.output_parsers import StrOutputParser
    from lib.prompt_templates import build_rag_prompt

    prompt_template = build_rag_prompt()
    llm = build_chat_model()
    return prompt_template | llm | StrOutputParser()

def has_usable_context(scored_documents):
    """Return True when retrieval produced at least one document with text."""

    if not scored_documents:
        return False

    return any(
        isinstance(document.page_content, str) and document.page_content.strip()
        for document, _distance in scored_documents
    )

def format_context(scored_documents):
    """Format retrieved LangChain documents into prompt-ready context text."""

    blocks = []
    for index, (document, distance) in enumerate(scored_documents, start=1):
        metadata = document.metadata or {}
        blocks.append(
            f"[Context {index}]\n"
            f"Source ID: {metadata.get('source_id', 'unknown')}\n"
            f"Title: {metadata.get('title', 'Untitled')}\n"
            f"Category: {metadata.get('category', 'Uncategorized')}\n"
            f"Section: {metadata.get('section', 'Unspecified')}\n"
            f"Chunk ID: {metadata.get('chunk_id', 'unknown')}\n"
            f"Distance: {float(distance):.4f}\n"
            f"Text: {document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the LangChain-supported RAG workflow for one validated question."""

    try:
        cleaned_question = question.strip()

        scored_documents = retrieve_context(
            cleaned_question, vector_store=vector_store, top_k=top_k
        )

        if not has_usable_context(scored_documents):
            debug = format_langchain_debug(scored_documents, "", top_k, True)
            return format_fallback_response(debug)

        context = format_context(scored_documents)

        if chain is None:
            chain = build_chain()

        answer = chain.invoke({"context": context, "question": cleaned_question})

        if not isinstance(answer, str) or not answer.strip():
            raise LangChainServiceError("The model returned an empty answer.")

        sources = format_sources(scored_documents)
        debug = format_langchain_debug(scored_documents, context, top_k, False)
        return format_success_response(answer, sources, debug)

    except LangChainServiceError:
        raise
    except Exception as exc:
        raise LangChainServiceError(str(exc)) from exc