"""
Production prompt templates with few-shot examples and chain-of-thought.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── System Prompts ──────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an expert document analysis assistant. Your role is to answer questions accurately based ONLY on the provided context documents.

## Instructions
1. Read the provided context documents carefully.
2. Answer the user's question based strictly on the context.
3. If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this question."
4. Cite specific sections or documents when possible.
5. Be concise but thorough.
6. If the question is ambiguous, ask for clarification.

## Rules
- NEVER fabricate information not present in the context.
- NEVER use prior knowledge — only the retrieved documents.
- If multiple documents provide conflicting information, note the discrepancy.
- Prefer direct quotes when the exact wording matters.

## Context Documents
{context}"""

CONDENSED_QUESTION_PROMPT = """Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full context needed for retrieval.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

# ── Prompt Templates ────────────────────────────────────────

def get_rag_prompt() -> ChatPromptTemplate:
    """Standard RAG prompt with conversation memory."""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ])


def get_rag_prompt_with_cot() -> ChatPromptTemplate:
    """RAG prompt with chain-of-thought reasoning."""
    cot_system = RAG_SYSTEM_PROMPT + """

## Reasoning Process
Before answering, think through these steps:
1. What specific information does the question ask for?
2. Which context documents are most relevant?
3. What is the most accurate and complete answer based on the evidence?
4. Are there any caveats or limitations to my answer?

Provide your reasoning briefly, then give the final answer."""

    return ChatPromptTemplate.from_messages([
        ("system", cot_system),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ])


def get_condense_question_prompt() -> ChatPromptTemplate:
    """Prompt for condensing follow-up questions with chat history."""
    return ChatPromptTemplate.from_template(CONDENSED_QUESTION_PROMPT)


def get_few_shot_rag_prompt() -> ChatPromptTemplate:
    """RAG prompt with few-shot examples for better formatting."""
    few_shot_system = RAG_SYSTEM_PROMPT + """

## Example Interaction

User: What is the company's revenue growth rate?
Assistant: Based on the Q3 2024 financial report (Document 2), the company achieved a revenue growth rate of 23% year-over-year, increasing from $4.2B to $5.17B. The report notes this was primarily driven by the cloud services division, which grew 41% (Document 2, Section 3.1).

User: How does that compare to competitors?
Assistant: The provided documents don't contain specific competitor revenue data. I can only speak to the company's own performance based on the available context. You may want to consult industry reports for competitive benchmarking."""

    return ChatPromptTemplate.from_messages([
        ("system", few_shot_system),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ])
