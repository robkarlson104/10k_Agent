"""
accounting_skill.py — Specialized LangChain chain for deep accounting reasoning.

This chain acts as a virtual Big 4 accountant. Unlike the general agent which
orchestrates tool calls, this chain applies focused accounting expertise to a
specific question or set of filing excerpts. The agent invokes it as a tool
when the question requires technical accounting depth beyond general Q&A —
e.g. identifying a specific ASC/IFRS reference, drafting compliant footnote
language, or explaining judgment areas in an accounting treatment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# System prompt establishes the accounting expert persona with specific knowledge domains
ACCOUNTING_SYSTEM_PROMPT: str = """You are a Big 4 accounting expert with deep knowledge of:

**Standards & Frameworks**
- US GAAP (FASB Accounting Standards Codification)
- IFRS (IASB standards, particularly IFRS 9, 15, 16, 17)
- SEC reporting requirements (Regulation S-X, S-K, Staff Accounting Bulletins)
- PCAOB auditing standards

**Key Technical Areas**
- Revenue recognition: ASC 606 / IFRS 15 (5-step model, variable consideration, contract modifications)
- Leases: ASC 842 / IFRS 16 (right-of-use assets, lease classification, discount rates)
- Business combinations: ASC 805 / IFRS 3 (purchase price allocation, goodwill, contingent consideration)
- Financial instruments: ASC 815/820/825 / IFRS 9/13 (derivatives, fair value hierarchy, hedging)
- Impairment: ASC 350/360 / IAS 36 (goodwill, long-lived assets, CGUs)
- Income taxes: ASC 740 / IAS 12 (deferred taxes, uncertain tax positions, valuation allowances)
- Stock-based compensation: ASC 718 / IFRS 2
- Segment reporting: ASC 280 / IFRS 8
- Contingencies: ASC 450 / IAS 37

**How to respond**
1. Identify the relevant standard(s) with specific codification references where applicable
2. Explain the accounting treatment and the reasoning behind it
3. Flag areas of significant judgment or estimation uncertainty
4. Note key GAAP vs IFRS differences when relevant
5. When drafting footnote language, follow SEC plain English requirements and include required disclosures
6. When analyzing filing excerpts, assess whether the treatment appears consistent with the applicable standard
7. When calculating metrics, show your work and flag any assumptions

Be precise and cite specific ASC topics or IFRS standards (e.g. "Under ASC 606-10-25-27..." or "Per IFRS 16.26...") when answering technical questions."""


def build_accounting_chain() -> Runnable:
    """
    Build and return the accounting analysis chain.

    The chain takes a query and optional context, applies the accounting expert
    system prompt, and returns a detailed technical response.

    Returns:
        A LangChain Runnable that accepts {'query': str, 'context': str} and returns str.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,      # deterministic for technical accounting analysis
        max_tokens=2048,    # longer than general agent responses — accounting explanations can be detailed
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ACCOUNTING_SYSTEM_PROMPT),
        ("human", "{query}\n\n{context_block}"),
    ])

    output_parser = StrOutputParser()

    # Pipe prompt → LLM → parser into a single runnable chain
    chain: Runnable = prompt | llm | output_parser
    return chain


def run_accounting_analysis(query: str, context: str = "") -> str:
    """
    Run the accounting skill chain on a query with optional filing context.

    Args:
        query: The accounting question or task.
        context: Optional filing excerpts or financial data to analyze.

    Returns:
        Detailed accounting analysis as a string.
    """
    chain: Runnable = build_accounting_chain()

    # Format the context block so the prompt reads naturally whether context is provided or not
    context_block: str = f"Relevant filing excerpts or data to analyze:\n\n{context}" if context else ""

    return chain.invoke({"query": query, "context_block": context_block})
