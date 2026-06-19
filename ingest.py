"""
ingest.py — Streaming pipeline that ingests S&P 500 10-K filings into pgvector.

Flow per ticker:
  1. Look up CIK from SEC ticker map
  2. Find most recent 10-K filing (2024 or 2025 filing date)
  3. Resolve the actual document URL from the filing index
  4. Download and parse HTML → section-aware text chunks
  5. Embed chunks with voyage-finance-2
  6. Store chunks + embeddings in Postgres
"""

import os
import io
import time
import requests
import pandas as pd
import voyageai
import psycopg2.extras
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from psycopg2.extensions import connection as PgConnection
from db import get_connection, init_db

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# SEC requires a descriptive User-Agent identifying who is making requests
SEC_HEADERS: dict[str, str] = {'User-Agent': os.getenv("SEC_EMAIL")}

voyage: voyageai.Client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

SP500_WIKI_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# voyage-finance-2 produces 1024-dimensional embeddings
EMBEDDING_DIM: int = 1024

# Word-based chunking — 800 words with 100-word overlap to preserve context at boundaries
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100

# SEC enforces a 10 requests/sec limit; 0.11s gap keeps us safely under it
RATE_LIMIT_DELAY: float = 0.11


def get_sp500_tickers() -> list[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.

    Returns a list of ticker symbols with dots replaced by dashes
    to match SEC EDGAR conventions (e.g. BRK.B → BRK-B).
    Uses requests to fetch the page so we can set a browser User-Agent
    (pd.read_html alone gets 403'd by Wikipedia).
    """
    resp = requests.get(
        SP500_WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; 10k-agent/1.0)"}
    )
    resp.raise_for_status()
    tables: list[pd.DataFrame] = pd.read_html(io.StringIO(resp.text))
    df: pd.DataFrame = tables[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def get_cik_map() -> dict[str, str]:
    """
    Download the SEC's full ticker-to-CIK mapping and return it as a dict.

    CIK numbers are zero-padded to 10 digits to match EDGAR URL formats.
    Keys are uppercased ticker symbols.
    """
    raw = requests.get("https://www.sec.gov/include/ticker.txt", headers=SEC_HEADERS)
    cik_map: dict[str, str] = {}
    for line in raw.text.strip().splitlines():
        parts: list[str] = line.split("\t")
        if len(parts) == 2:
            cik_map[parts[0].upper()] = parts[1].zfill(10)
    return cik_map


def get_10k_filing(cik: str) -> dict | None:
    """
    Fetch a company's recent filings from EDGAR and return the most recent 10-K
    filed in 2024 or 2025.

    Returns a dict with 'date' and 'doc_url' keys, or None if not found.
    The primaryDocument field in the submissions JSON lets us build the
    document URL directly without a second request to the index page.
    """
    url: str = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(RATE_LIMIT_DELAY)
    resp = requests.get(url, headers=SEC_HEADERS)
    if resp.status_code != 200:
        return None

    data: dict = resp.json()
    filings: dict = data.get("filings", {}).get("recent", {})
    forms: list[str] = filings.get("form", [])
    dates: list[str] = filings.get("filingDate", [])
    accessions: list[str] = filings.get("accessionNumber", [])
    primary_docs: list[str] = filings.get("primaryDocument", [])

    for form, date, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
        if form == "10-K" and (date.startswith("2024") or date.startswith("2025")):
            # Folder uses accession without dashes; filename is the primaryDocument
            accession_no_dashes: str = accession.replace("-", "")
            doc_url: str = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
                f"/{accession_no_dashes}/{primary_doc}"
            )
            return {"date": date, "doc_url": doc_url}
    return None


def parse_sections(html: str) -> list[dict[str, str]]:
    """
    Extract named sections from a 10-K HTML document.

    Strips HTML tags and splits the plain text at standard Item headings
    (Item 1 through Item 15). Returns a list of dicts with 'section' and 'text' keys.
    Content before the first Item heading is labeled 'Preamble'.
    """
    soup = BeautifulSoup(html, "lxml")
    text: str = soup.get_text(separator="\n", strip=True)

    # Standard 10-K section headings defined by SEC Regulation S-K
    section_markers: list[str] = [
        "Item 1.", "Item 1A.", "Item 1B.", "Item 2.", "Item 3.",
        "Item 4.", "Item 5.", "Item 6.", "Item 7.", "Item 7A.",
        "Item 8.", "Item 9.", "Item 9A.", "Item 10.", "Item 11.",
        "Item 12.", "Item 13.", "Item 14.", "Item 15.",
    ]

    sections: list[dict[str, str]] = []
    current_section: str = "Preamble"
    current_lines: list[str] = []

    for line in text.splitlines():
        matched = next((m for m in section_markers if line.strip().startswith(m)), None)
        if matched:
            if current_lines:
                sections.append({"section": current_section, "text": "\n".join(current_lines)})
            current_section = matched
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"section": current_section, "text": "\n".join(current_lines)})

    return sections


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Uses CHUNK_SIZE words per chunk with CHUNK_OVERLAP words of overlap
    so that context is preserved across chunk boundaries.
    """
    words: list[str] = text.split()
    chunks: list[str] = []
    start: int = 0
    while start < len(words):
        end: int = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks using Voyage AI's voyage-finance-2 model.

    Batches requests at 64 chunks at a time to stay within Voyage's rate limits.
    Returns a list of 1024-dimensional float vectors, one per chunk.
    """
    all_embeddings: list[list[float]] = []
    batch_size: int = 64
    for i in range(0, len(chunks), batch_size):
        batch: list[str] = chunks[i:i + batch_size]
        result = voyage.embed(batch, model="voyage-finance-2", input_type="document")
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def store_chunks(
    conn: PgConnection,
    ticker: str,
    cik: str,
    filed_date: str,
    section: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """
    Bulk-insert a set of chunks and their embeddings into the filings table.

    Uses execute_values for efficient batch insertion.
    The embedding is cast to the vector type expected by pgvector.
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO filings (ticker, cik, filed_date, section, chunk_index, content, embedding)
            VALUES %s
            """,
            [
                (ticker, cik, filed_date, section, i, chunk, embedding)
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
            template="(%s, %s, %s, %s, %s, %s, %s::vector)"
        )
    conn.commit()


def process_ticker(ticker: str, cik: str, conn: PgConnection) -> None:
    """
    Run the full ingest pipeline for a single ticker.

    Fetches the 10-K filing, downloads and parses it, chunks each section,
    embeds the chunks, and stores them in Postgres. Skips gracefully if any
    step fails (no filing found, doc URL unresolvable, download error).
    """
    print(f"  [{ticker}] fetching filings...")
    filing: dict | None = get_10k_filing(cik)
    if not filing:
        print(f"  [{ticker}] no 2024/2025 10-K found, skipping.")
        return

    doc_url: str = filing["doc_url"]
    print(f"  [{ticker}] downloading {doc_url}")
    time.sleep(RATE_LIMIT_DELAY)
    resp = requests.get(doc_url, headers=SEC_HEADERS)
    if resp.status_code != 200:
        print(f"  [{ticker}] download failed ({resp.status_code}), skipping.")
        return

    sections: list[dict[str, str]] = parse_sections(resp.text)
    total_chunks: int = 0

    for sec in sections:
        chunks: list[str] = chunk_text(sec["text"])
        if not chunks:
            continue
        embeddings: list[list[float]] = embed_chunks(chunks)
        store_chunks(conn, ticker, cik, filing["date"], sec["section"], chunks, embeddings)
        total_chunks += len(chunks)

    print(f"  [{ticker}] done — {total_chunks} chunks stored.")


def run() -> None:
    """
    Entry point for the full S&P 500 ingest run.

    Initializes the database, fetches tickers and the CIK map, then
    streams through each ticker one at a time. Errors on individual tickers
    are caught and logged so the run continues.
    """
    init_db()
    print("Fetching S&P 500 tickers...")
    tickers: list[str] = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers.")

    print("Building CIK map...")
    cik_map: dict[str, str] = get_cik_map()

    conn: PgConnection = get_connection()
    try:
        for ticker in tickers:
            cik: str | None = cik_map.get(ticker.upper())
            if not cik:
                print(f"  [{ticker}] no CIK found, skipping.")
                continue
            try:
                process_ticker(ticker, cik, conn)
            except Exception as e:
                print(f"  [{ticker}] error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
