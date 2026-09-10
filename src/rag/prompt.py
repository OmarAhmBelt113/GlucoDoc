from langchain_core.documents import Document

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are GlucoDoc, a medical information assistant specialized in diabetes.

Your task is to answer the user's question using ONLY the information provided in the retrieved context.
You also have access to the conversation history to maintain context for follow-up questions.

STRICT RULES:

1. Ground every factual claim in the retrieved context.

2. Do not invent, assume, infer, or add medical information that is not explicitly supported by the context.

3. If the retrieved context does not contain enough information to answer the question, clearly state that the available sources do not provide enough information.

4. Do not make a definitive medical diagnosis for the user.

5. Do not prescribe medications or provide personalized treatment plans.

6. When discussing diagnostic criteria, laboratory values, thresholds, or measurements, preserve the values and units exactly as they appear in the retrieved context.

7. Prefer clear, concise, and easy-to-understand explanations.

8. When multiple retrieved sources are relevant, synthesize their information without introducing unsupported claims.

9. The retrieved context is the ONLY source of truth.

10. IMPORTANT: You MUST cite your sources using inline citations. When you make a factual claim based on a context item, append an inline citation in the format [Source Name, Page X] immediately after the claim. For example: "The diagnostic criteria for diabetes is a fasting plasma glucose of 126 mg/dL [diabetes_guidelines.pdf, Page 12]."

11. Do not include a "Sources", "References", "Citations", or "Resources" list at the end of your answer. The citations must be inline.
"""

# ============================================================
# Context Formatting
# ============================================================

def format_context(documents: list[Document]) -> str:
    """
    Format retrieved documents into structured context for the LLM.
    """
    if not documents:
        return "No relevant context was retrieved."

    formatted_context = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata if document.metadata else {}
        file_name = metadata.get("file_name", metadata.get("source", "Unknown source"))
        page = metadata.get("page", metadata.get("page_number", "Unknown page"))
        content = (document.page_content or "").strip()

        formatted_context.append(
            f"[Context {index}]\nSource: {file_name}\nPage: {page}\nContent:\n{content}\n"
        )

    return "\n".join(formatted_context)


def format_history(history: list) -> str:
    """
    Format conversation history for the LLM.
    """
    if not history:
        return "No previous conversation."
    
    formatted = []
    for msg in history:
        # Check if msg is a dict (if we receive a dict) or an object with role/content
        if isinstance(msg, dict):
            role = str(msg.get("role", "Unknown")).capitalize()
            content = str(msg.get("content", ""))
        else:
            role = str(msg.role).capitalize()
            content = str(msg.content)
            
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)

# ============================================================
# Prompt Builder
# ============================================================

def build_prompt(
    question: str,
    documents: list[Document],
    history: list = None,
    summarize: bool = False,
) -> str:
    """
    Build the final prompt sent to Gemini.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    context = format_context(documents=documents)
    history_str = format_history(history)

    rules = """
Answer the user's question using ONLY the retrieved context.

Your response must:
- Directly answer the question.
- Be concise and clear.
- Preserve medical values and units exactly when relevant.
- Avoid unsupported claims.
- Avoid diagnosis or personalized treatment.
- ALWAYS include inline citations like [Source Name, Page X] for factual claims.
- NOT contain a Sources/References/Resources section at the end.
"""
    if summarize:
        rules += "\n- PROVIDE A SHORT, HIGH-LEVEL SUMMARY instead of a detailed explanation.\n"

    prompt = f"""
{SYSTEM_PROMPT}

================ CONVERSATION HISTORY ================

{history_str}

================ RETRIEVED CONTEXT ================

{context}

================ USER QUESTION ================

{question.strip()}

================ FINAL RESPONSE RULES ================
{rules.strip()}
"""

    return prompt.strip()
