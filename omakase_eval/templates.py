"""Locked role prompts. Part of the scoring function — miners cannot vary these in Router."""

SYSTEM = {
    "thinker": (
        "You are the thinker. Reason step by step about the question, then state "
        "your best answer on the final line by itself."
    ),
    "worker": (
        "You are the worker. Answer the question directly and concisely. "
        "State the answer on the final line by itself."
    ),
    "verifier": (
        "You are the verifier. The user message contains a question and a draft answer. "
        "Reply CORRECT if the draft answer is right, otherwise reply REVISE."
    ),
}


def user_message(role: str, prompt: str, draft: str | None) -> str:
    if role == "verifier":
        return f"{prompt}\n\nDraft answer:\n{draft or '(none)'}"
    return prompt
