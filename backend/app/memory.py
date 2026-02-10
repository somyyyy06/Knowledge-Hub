conversation_store = {}

def get_history(session_id: str):
    """
    Returns the conversation history for a session.
    """
    return conversation_store.get(session_id, [])

def add_to_history(session_id: str, role: str, content: str):
    """
    Adds a message to the conversation history.
    role: 'user' or 'assistant'
    """
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append({
        "role": role,
        "content": content
    })