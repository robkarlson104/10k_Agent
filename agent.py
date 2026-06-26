"""
agent.py — LangChain agent for querying S&P 500 10-K filings.

Uses LangChain 1.x create_agent (tool-calling loop) backed by LangGraph.
Conversation memory is provided via MemorySaver — all turns in the same
session share a thread_id so the agent can reference prior context.

Tools available to the agent:
  - search_filings: semantic search across all filings (optional ticker/section filter)
  - compare_companies: side-by-side accounting treatment comparison
  - get_sector_practices: how a GICS sector handles a specific topic
  - search_accounting_standards: search GAAP/IFRS standards via DuckDuckGo
  - accounting_analysis: deep accounting reasoning chain (Big 4 expert persona)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import TOOLS
from audit import init_audit_tables, create_session, close_session, AuditCallbackHandler

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

SYSTEM_PROMPT: str = """You are an expert financial analyst assistant specializing in SEC 10-K filings.
You have access to a database of S&P 500 10-K filings from 2024-2025.

Your capabilities:
- Answer questions about specific companies' financial performance, risks, and strategies
- Identify and explain accounting treatments and policies with specific standard references
- Compare how different companies or sectors handle specific accounting topics
- Help draft or review footnote language based on peer company disclosures and applicable standards
- Pull relevant financial metrics and disclosures from filings
- Apply deep accounting expertise (GAAP/IFRS) to analyze treatments in filing excerpts

How to approach questions:
- For company-specific questions, use search_filings with the ticker filter
- For comparison questions, use compare_companies to pull data side by side
- For sector-wide questions, use get_sector_practices to find industry norms
- For standard-specific questions, use search_accounting_standards for GAAP or IFRS guidance
- For technical accounting analysis, use accounting_analysis — pass filing excerpts in the context field
- Chain tools together when needed: search first, then pass results to accounting_analysis for deeper reasoning
- Always cite the specific company, filing date, and section when referencing filing information
- Always cite the specific ASC topic or IFRS standard when referencing accounting guidance"""


def build_agent():
    """
    Assemble and return a LangChain 1.x agent graph with conversation memory.

    Memory is persisted via MemorySaver checkpointer — each session uses a
    thread_id so the agent retains context across turns within that session.

    Returns:
        A compiled StateGraph ready to invoke.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
        max_tokens=4096,
    )

    init_audit_tables()

    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


def run_agent(question: str, agent, thread_id: str = "default") -> str:
    """
    Stream a question through the agent, printing tool calls and observations
    as they happen, then return the final answer.

    Args:
        question: The user's natural language question about 10-K filings.
        agent: A compiled agent graph from build_agent().
        thread_id: Session identifier for conversation continuity.

    Returns:
        The agent's final answer as a string.
    """
    session_id = create_session(thread_id, question)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [AuditCallbackHandler(session_id)],
    }
    final_answer = ""

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        for node, updates in chunk.items():
            messages = updates.get("messages", [])
            for msg in messages:
                # Tool call being made
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"[Calling tool: {tc['name']}]")
                # Tool result coming back
                elif hasattr(msg, "type") and msg.type == "tool":
                    print(f"[Tool result received from: {msg.name}]")
                # Final AI response — capture any non-tool AI message
                elif hasattr(msg, "content") and msg.content:
                    if getattr(msg, "type", "") not in ("tool", "human"):
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            final_answer = msg.content

    close_session(session_id, final_answer)
    return final_answer


if __name__ == "__main__":
    print("Building 10-K analyst agent...")
    executor = build_agent()
    print("Agent ready. Type 'quit' to exit.\n")

    while True:
        question: str = input("Question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        print("\n" + "=" * 60)
        answer: str = run_agent(question, executor)
        print("=" * 60)
        print(f"\nFinal Answer:\n{answer}\n")
