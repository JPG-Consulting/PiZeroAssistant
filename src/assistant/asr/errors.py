class ASRError(Exception):
    """Base exception for all ASR-related errors."""


class ASRTimeoutError(ASRError):
    """ASR request timed out."""


class ASRUnavailableError(ASRError):
    """ASR service is unreachable or unavailable."""


class ASRAuthenticationError(ASRError):
    """Invalid credentials or authentication failure."""
