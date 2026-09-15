"""Shape expected retrieval and generation errors for HTTP responses."""


def error_detail(error: Exception) -> str:
    """Format an expected boundary failure without implementation details."""
    if isinstance(error, FileNotFoundError):
        missing_path = error.filename or str(error)
        return f"File not found: {missing_path}"
    if isinstance(error, NotADirectoryError):
        invalid_path = error.filename or str(error)
        return f"Directory not found: {invalid_path}"
    return str(error) or error.__class__.__name__
