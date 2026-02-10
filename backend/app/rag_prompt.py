def build_rag_prompt(context_chunks, question, chat_history=None):
    prompt = """You are a helpful assistant.

You MUST answer the question using ONLY the context provided.
If the answer is not present, say:
"I don't know based on the provided documents."

"""

    if chat_history:
        prompt += "Conversation history:\n"
        for turn in chat_history[-6:]:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
        prompt += "\n"

    prompt += "Context:\n"

    for chunk in context_chunks:
        prompt += f"- {chunk}\n"

    prompt += f"\nQuestion: {question}\nAnswer:"

    return prompt
