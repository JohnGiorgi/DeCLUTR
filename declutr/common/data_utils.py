def sanitize(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    if lowercase:
        return sanitized_text.lower()
    else:
        return sanitized_text
