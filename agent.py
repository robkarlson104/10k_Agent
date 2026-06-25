"""
agent.py — LangChain ReAct agent for querying S&P 500 10-K filings.

The agent uses a ReAct (Reasoning + Acting) loop to decide which tools to call
and in what order. Conversation memory lets it reference earlier turns in the
same session — e.g. "compare that to what you found for Apple" or "now look at
the IFRS treatment for the same topic."

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
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from tools import TOOLS

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Prompt template — must include {tools}, {tool_names}, {chat_history}, {input}, {agent_scratchpad}
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
- Always cite the specific ASC topic or IFRS standard when referencing accounting guidance

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, with citations

Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def build_agent() -> AgentExecutor:
    """
    Assemble and return a LangChain AgentExecutor with conversation memory.

    Memory is a sliding window of the last 10 exchanges so the agent can
    reference prior turns without the context growing unbounded.
    handle_parsing_errors recovers gracefully if the LLM output format slips.

    Returns:
        A ready-to-invoke AgentExecutor with memory attached.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
        max_tokens=4096,
    )

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
    )

    # Keep the last 10 conversation turns in memory — enough for a session
    # without blowing up the context window on long research sessions
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,
        return_messages=False,  # plain text format, compatible with the PromptTemplate above
    )

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )


def run_agent(question: str, agent_executor: AgentExecutor) -> str:
    """
    Run a single question through the agent and return the final answer.

    Memory is stored on the AgentExecutor instance, so prior turns are
    automatically included in each subsequent invocation.

    Args:
        question: The user's natural language question about 10-K filings.
        agent_executor: A pre-built AgentExecutor from build_agent().

    Returns:
        The agent's final synthesized answer as a string.
    """
    result: dict = agent_executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    print("Building 10-K analyst agent...")
    executor: AgentExecutor = build_agent()
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
