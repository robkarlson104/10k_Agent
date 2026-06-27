"""
schemas.py — Pydantic models for agent tool inputs and outputs.

All tool inputs are validated through these schemas before execution,
giving us type safety and clear documentation of what each tool expects.
"""

from pydantic import BaseModel, Field


class SearchFilingsInput(BaseModel):
    """Input schema for semantic search across ingested 10-K filings."""

    query: str = Field(
        ...,
        description="Natural language question about 10-K filings, accounting treatments, or financial metrics.",
    )
    ticker: str | None = Field(
        None,
        description="Optional ticker symbol to scope the search to one company (e.g. 'AAPL').",
    )
    section: str | None = Field(
        None,
        description="Optional 10-K section to filter by (e.g. 'Item 7.', 'Item 1A.', 'Item 8.').",
    )
    n_results: int = Field(
        8, description="Number of filing chunks to retrieve. Default is 8."
    )


class CompareCompaniesInput(BaseModel):
    """Input schema for comparing how multiple companies handle a specific topic."""

    query: str = Field(
        ...,
        description="Accounting topic, metric, or treatment to compare across companies.",
    )
    tickers: list[str] = Field(
        ...,
        description="List of ticker symbols to compare (e.g. ['AAPL', 'MSFT', 'GOOGL']).",
    )
    n_results_per_company: int = Field(
        3, description="Number of chunks to retrieve per company. Default is 3."
    )


class SectorPracticesInput(BaseModel):
    """Input schema for finding how companies in a sector handle an accounting topic."""

    query: str = Field(
        ..., description="Accounting topic or practice to research across the sector."
    )
    sector: str = Field(
        ...,
        description=(
            "GICS sector name. Valid values: 'Communication Services', 'Consumer Discretionary', "
            "'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials', "
            "'Information Technology', 'Materials', 'Real Estate', 'Utilities'."
        ),
    )
    n_results: int = Field(
        10,
        description="Total number of chunks to retrieve across all companies in the sector.",
    )


class SearchAccountingStandardsInput(BaseModel):
    """Input schema for searching GAAP and/or IFRS accounting standards."""

    query: str = Field(
        ...,
        description="Accounting topic, treatment, or standard to look up (e.g. 'lease accounting', 'revenue recognition', 'goodwill impairment').",
    )
    standard: str = Field(
        "both",
        description="Which standard to search: 'GAAP', 'IFRS', or 'both'. Defaults to 'both'.",
    )
    n_results: int = Field(
        5, description="Number of search results to return. Default is 5."
    )


class AccountingAnalysisInput(BaseModel):
    """Input schema for the deep accounting skill chain."""

    query: str = Field(
        ...,
        description=(
            "The accounting question or task. Examples: 'identify the revenue recognition treatment', "
            "'draft a lease liability footnote', 'explain the GAAP vs IFRS difference for goodwill impairment', "
            "'calculate the debt-to-equity ratio from these figures'."
        ),
    )
    context: str = Field(
        "",
        description="Optional filing excerpts, financial figures, or other content to analyze. Leave empty for conceptual questions.",
    )


class FilingChunk(BaseModel):
    """A single retrieved chunk from the filings table, used in tool outputs."""

    ticker: str
    filed_date: str
    section: str
    content: str
