"""Prompt template helpers for the LangChain RAG workflow."""

SYSTEM_PROMPT = """
You are an internal reliability assistant.

Answer using only the approved retrieved context provided by the backend.
If the context does not contain enough information to answer reliably, say that
you do not have enough approved context to answer. Do not invent policies,
procedures, metrics, or incident steps.

Keep the response concise, specific, and useful to an engineer during an incident.
"""


def build_rag_prompt():

    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Approved context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer using only the approved context above.",
            ),
        ]
    )