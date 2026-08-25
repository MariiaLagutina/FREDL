def validate(value: float) -> bool:
    """Apply request timeout validation before sending."""
    return value > 0
