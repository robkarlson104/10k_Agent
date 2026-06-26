"""
tools.py — LangChain tools that wrap pgvector retrieval for the 10-K agent.

Each tool is a structured LangChain tool backed by a Pydantic input schema.
The agent decides which tool to call and with what arguments based on the
user's question. Tools are composable — the agent can call multiple in sequence.
"""

import os
import io
import time
import requests
import voyageai
import pandas as pd
import psycopg2.extras
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from duckduckgo_search import DDGS
from schemas import SearchFilingsInput, CompareCompaniesInput, SectorPracticesInput, SearchAccountingStandardsInput, AccountingAnalysisInput, FilingChunk
from accounting_skill import run_accounting_analysis
from audit import log_db_query
from db import get_connection

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

voyage: voyageai.Client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

SP500_WIKI_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Module-level cache so we only fetch the sector map once per session
_sector_map: dict[str, str] | None = None


def _get_sector_map() -> dict[str, str]:
    """
    Fetch and cache a mapping of ticker → GICS sector from Wikipedia.

    Lazy-loaded on first use and cached for the rest of the session.
    Returns a dict like {'AAPL': 'Information Technology', 'JPM': 'Financials', ...}.
    """
    global _sector_map
    if _sector_map is not None:
        return _sector_map

    resp = requests.get(
        SP500_WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; 10k-agent/1.0)"}
    )
    resp.raise_for_status()
    tables: list[pd.DataFrame] = pd.read_html(io.StringIO(resp.text))
    df: pd.DataFrame = tables[0]

    # Normalize tickers to match EDGAR format (dots → dashes)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    _sector_map = dict(zip(df["Symbol"].str.upper(), df["GICS Sector"]))
    return _sector_map


def _embed_query(query: str) -> list[float]:
    """
    Embed a user query using Voyage AI with input_type='query'.

    Query embeddings are optimized for retrieval (vs 'document' used during ingest).
    Returns a 1024-dimensional float vector.
    """
    result = voyage.embed([query], model="voyage-finance-2", input_type="query")
    return result.embeddings[0]


def _format_chunks(rows: list[tuple]) -> str:
    """
    Format raw database rows into a readable string for the agent.

    Each chunk is labeled with its ticker, date, and section so the agent
    can cite sources accurately in its response.
    """
    if not rows:
        return "No relevant filing content found."

    parts: list[str] = []
    for ticker, filed_date, section, content in rows:
        parts.append(f"[{ticker} | {filed_date} | {section}]\n{content}")
    return "\n\n---\n\n".join(parts)


def search_filings(query: str, ticker: str | None = None, section: str | None = None, n_results: int = 8) -> str:
    """
    Perform a semantic search across all ingested 10-K filings.

    Embeds the query with Voyage AI and finds the most semantically similar
    chunks in pgvector using cosine distance. Optionally filter by ticker
    or 10-K section to narrow results.

    Args:
        query: Natural language question about filings or accounting topics.
        ticker: Optional company ticker to restrict search to one company.
        section: Optional 10-K section name to restrict search (e.g. 'Item 7.').
        n_results: Number of chunks to return.

    Returns:
        Formatted string of relevant filing excerpts with source labels.
    """
    embedding: list[float] = _embed_query(query)

    sql: str
    params: tuple

    if ticker and section:
        sql = """
            SELECT ticker, filed_date, section, content
            FROM filings
            WHERE ticker = %s AND section = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s"""
        params = (ticker.upper(), section, embedding, n_results)
    elif ticker:
        sql = """
            SELECT ticker, filed_date, section, content
            FROM filings
            WHERE ticker = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s"""
        params = (ticker.upper(), embedding, n_results)
    elif section:
        sql = """
            SELECT ticker, filed_date, section, content
            FROM filings
            WHERE section = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s"""
        params = (section, embedding, n_results)
    else:
        sql = """
            SELECT ticker, filed_date, section, content
            FROM filings
            ORDER BY embedding <=> %s::vector
            LIMIT %s"""
        params = (embedding, n_results)

    with get_connection() as conn:
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(sql, params)
            rows = cur.fetchall()
            duration_ms = int((time.perf_counter() - t0) * 1000)

    log_db_query(
        sql,
        {"query": query, "ticker": ticker, "section": section, "n_results": n_results},
        len(rows),
        "search_filings",
        duration_ms,
    )
    return _format_chunks(rows)


def compare_companies(query: str, tickers: list[str], n_results_per_company: int = 3) -> str:
    """
    Compare how multiple companies handle a specific accounting topic or metric.

    Runs a separate semantic search for each ticker and assembles results
    side by side so the agent can identify similarities and differences
    in accounting treatments across companies.

    Args:
        query: Accounting topic or metric to compare.
        tickers: List of ticker symbols to compare.
        n_results_per_company: Chunks to retrieve per company.

    Returns:
        Formatted string grouping results by company.
    """
    embedding: list[float] = _embed_query(query)
    all_parts: list[str] = []
    sql = """
        SELECT ticker, filed_date, section, content
        FROM filings
        WHERE ticker = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s"""

    with get_connection() as conn:
        with conn.cursor() as cur:
            for ticker in tickers:
                t0 = time.perf_counter()
                cur.execute(sql, (ticker.upper(), embedding, n_results_per_company))
                rows = cur.fetchall()
                duration_ms = int((time.perf_counter() - t0) * 1000)
                log_db_query(
                    sql,
                    {"query": query, "ticker": ticker, "n_results": n_results_per_company},
                    len(rows),
                    "compare_companies",
                    duration_ms,
                )
                if rows:
                    all_parts.append(f"=== {ticker.upper()} ===\n{_format_chunks(rows)}")
                else:
                    all_parts.append(f"=== {ticker.upper()} ===\nNo data found for this ticker.")

    return "\n\n".join(all_parts)


def get_sector_practices(query: str, sector: str, n_results: int = 10) -> str:
    """
    Find how companies in a specific GICS sector handle an accounting topic.

    Looks up all S&P 500 tickers in the given sector, then runs a semantic
    search scoped to those tickers. Useful for questions like 'how do
    Financial companies disclose credit risk?' or 'how does the Tech sector
    handle R&D capitalization?'.

    Args:
        query: Accounting topic or practice to research.
        sector: GICS sector name (e.g. 'Information Technology', 'Financials').
        n_results: Total chunks to retrieve across all sector companies.

    Returns:
        Formatted string of relevant excerpts from sector companies.
    """
    sector_map: dict[str, str] = _get_sector_map()

    # Find tickers that belong to the requested sector
    sector_tickers: list[str] = [
        ticker for ticker, s in sector_map.items()
        if s.lower() == sector.lower()
    ]

    if not sector_tickers:
        return f"No tickers found for sector '{sector}'. Check the sector name is a valid GICS sector."

    embedding: list[float] = _embed_query(query)
    sql = """
        SELECT ticker, filed_date, section, content
        FROM filings
        WHERE ticker = ANY(%s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s"""

    with get_connection() as conn:
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(sql, (sector_tickers, embedding, n_results))
            rows = cur.fetchall()
            duration_ms = int((time.perf_counter() - t0) * 1000)

    log_db_query(
        sql,
        {"query": query, "sector": sector, "sector_ticker_count": len(sector_tickers), "n_results": n_results},
        len(rows),
        "get_sector_practices",
        duration_ms,
    )
    header: str = f"Results from {len(sector_tickers)} '{sector}' sector companies:\n\n"
    return header + _format_chunks(rows)


def search_accounting_standards(query: str, standard: str = "both", n_results: int = 5) -> str:
    """
    Search GAAP and/or IFRS accounting standards using DuckDuckGo scoped to authoritative sources.

    Searches ifrs.org for IFRS guidance and fasb.org for GAAP/ASC guidance.
    Falls back to a broader accounting search if site-scoped results are sparse.
    Useful when a user asks how a topic *should* be handled under a standard,
    as opposed to how a specific company *does* handle it in their filings.

    Args:
        query: Accounting topic or treatment to look up.
        standard: 'GAAP', 'IFRS', or 'both'.
        n_results: Number of results to return.

    Returns:
        Formatted string of search results with titles, URLs, and snippets.
    """
    standard_upper: str = standard.upper()

    # Build a site-scoped query for the authoritative standard body
    if standard_upper == "IFRS":
        scoped_query: str = f"site:ifrs.org {query}"
    elif standard_upper == "GAAP":
        scoped_query = f"site:fasb.org {query}"
    else:
        scoped_query = f"(site:ifrs.org OR site:fasb.org) {query}"

    results: list[dict] = []
    with DDGS() as ddgs:
        results = list(ddgs.text(scoped_query, max_results=n_results))

    # Fall back to a broader search if the site-scoped query returns nothing
    if not results:
        fallback_label: str = "IFRS" if standard_upper == "IFRS" else "GAAP ASC"
        fallback_query: str = f"{query} {fallback_label} accounting standard guidance"
        with DDGS() as ddgs:
            results = list(ddgs.text(fallback_query, max_results=n_results))

    if not results:
        return f"No results found for '{query}' under {standard}. Try rephrasing the query."

    parts: list[str] = []
    for r in results:
        title: str = r.get("title", "No title")
        url: str = r.get("href", "")
        snippet: str = r.get("body", "No content available")
        parts.append(f"**{title}**\n{url}\n{snippet}")

    header: str = f"[{standard.upper()} Standards Search: '{query}']\n\n"
    return header + "\n\n---\n\n".join(parts)


# --- LangChain tool definitions ---
# Each StructuredTool wraps a function with its Pydantic schema so LangChain
# can validate inputs and present the tool correctly to the agent.

search_filings_tool = StructuredTool.from_function(
    func=search_filings,
    name="search_filings",
    description=(
        "Search across all ingested S&P 500 10-K filings using semantic similarity. "
        "Use this to answer questions about specific companies, accounting policies, "
        "risk disclosures, financial metrics, or footnote language. "
        "Optionally filter by ticker symbol or 10-K section (e.g. 'Item 7.', 'Item 1A.')."
    ),
    args_schema=SearchFilingsInput,
)

compare_companies_tool = StructuredTool.from_function(
    func=compare_companies,
    name="compare_companies",
    description=(
        "Compare how multiple companies handle a specific accounting topic, metric, or disclosure. "
        "Use this when the user wants to see differences or similarities across companies — "
        "e.g. 'how do Apple and Microsoft account for stock compensation?' or "
        "'compare revenue recognition policies across these three companies'."
    ),
    args_schema=CompareCompaniesInput,
)

get_sector_practices_tool = StructuredTool.from_function(
    func=get_sector_practices,
    name="get_sector_practices",
    description=(
        "Find how companies in a specific industry sector handle an accounting topic or disclosure. "
        "Use this for sector-level questions like 'how do banks disclose credit losses?' or "
        "'how does the healthcare sector handle contingent liabilities?'. "
        "Valid sectors: 'Communication Services', 'Consumer Discretionary', 'Consumer Staples', "
        "'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology', "
        "'Materials', 'Real Estate', 'Utilities'."
    ),
    args_schema=SectorPracticesInput,
)

search_accounting_standards_tool = StructuredTool.from_function(
    func=search_accounting_standards,
    name="search_accounting_standards",
    description=(
        "Search GAAP (FASB) and/or IFRS standards for authoritative accounting guidance. "
        "Use this when the user asks how something *should* be accounted for under a standard, "
        "wants to know the rules behind a treatment, needs to understand differences between "
        "GAAP and IFRS, or wants to draft footnote language that complies with a standard. "
        "Specify standard='GAAP', 'IFRS', or 'both'."
    ),
    args_schema=SearchAccountingStandardsInput,
)

accounting_analysis_tool = StructuredTool.from_function(
    func=run_accounting_analysis,
    name="accounting_analysis",
    description=(
        "Apply deep accounting expertise to a question or set of filing excerpts. "
        "Use this when the question requires technical depth beyond general Q&A — e.g. "
        "identifying the specific ASC or IFRS standard that applies, explaining a complex "
        "accounting treatment, drafting compliant footnote language, calculating financial "
        "ratios or metrics from provided figures, or assessing whether a company's disclosed "
        "treatment is consistent with the applicable standard. "
        "Pass relevant filing excerpts in the 'context' field when analyzing specific company disclosures."
    ),
    args_schema=AccountingAnalysisInput,
)

# Exported list of all tools for the agent to use
TOOLS: list[StructuredTool] = [
    search_filings_tool,
    compare_companies_tool,
    get_sector_practices_tool,
    search_accounting_standards_tool,
    accounting_analysis_tool,
]
