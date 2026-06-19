"""
rag.py — Query layer for the 10-K RAG system.

Retrieves semantically relevant filing chunks from pgvector using
Voyage AI embeddings, then passes them as context to Claude for generation.
"""

import os
import voyageai
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from db import get_connection

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

voyage: voyageai.Client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
claude: anthropic.Anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Number of chunks to retrieve from pgvector per query
TOP_K: int = 8


def retrieve(query: str, ticker: str | None = None) -> list[dict]:
    """
    Embed the query and retrieve the TOP_K most semantically similar chunks from Postgres.

    If ticker is provided, search is scoped to that company only.
    Uses cosine distance (<=> operator) via the hnsw index for fast ANN search.

    Returns a list of dicts with keys: ticker, filed_date, section, content.
    """
    # Embed the query with input_type="query" (vs "document" used during ingest)
    result = voyage.embed([query], model="voyage-finance-2", input_type="query")
    query_embedding: list[float] = result.embeddings[0]

    with get_connection() as conn:
        with conn.cursor() as cur:
            if ticker:
                cur.execute(
                    """
                    SELECT ticker, filed_date, section, content
                    FROM filings
                    WHERE ticker = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (ticker.upper(), query_embedding, TOP_K)
                )
            else:
                cur.execute(
                    """
                    SELECT ticker, filed_date, section, content
                    FROM filings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, TOP_K)
                )
            rows = cur.fetchall()

    return [
        {"ticker": r[0], "filed_date": str(r[1]), "section": r[2], "content": r[3]}
        for r in rows
    ]


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a labeled context block for the Claude prompt.

    Each chunk is prefixed with its ticker, filing date, and section name
    so Claude can cite sources in its response.
    """
    parts: list[str] = []
    for c in chunks:
        parts.append(f"[{c['ticker']} | {c['filed_date']} | {c['section']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def query(question: str, ticker: str | None = None) -> str:
    """
    Run a full RAG query: retrieve relevant chunks, build a prompt, and generate with Claude.

    Args:
        question: The user's natural language question about 10-K filings.
        ticker: Optional ticker symbol to scope the search to one company.

    Returns:
        Claude's response as a string, citing specific tickers and sections.
    """
    chunks: list[dict] = retrieve(question, ticker=ticker)
    if not chunks:
        return "No relevant filings found. Make sure ingest has been run."

    context: str = build_context(chunks)
    prompt: str = f"""You are a financial analyst assistant. Use the 10-K filing excerpts below to answer the question.
Cite the company ticker and section when referencing specific information.

<excerpts>
{context}
</excerpts>

Question: {question}"""

    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


if __name__ == "__main__":
    print("10-K RAG ready. Type 'quit' to exit.")
    while True:
        ticker_input: str | None = input("\nFilter by ticker (or press Enter for all): ").strip() or None
        question: str = input("Question: ").strip()
        if question.lower() == "quit":
            break
        print("\n" + query(question, ticker=ticker_input))
