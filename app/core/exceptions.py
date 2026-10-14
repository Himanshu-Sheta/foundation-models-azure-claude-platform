class FoundationModelError(Exception):
    """Base exception for all foundation model platform errors."""
    def __init__(self, message: str, provider: str = "unknown", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code


class ProviderAuthenticationError(FoundationModelError):
    """Raised when an API key or Azure token is invalid."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, status_code=401)


class RateLimitExceededError(FoundationModelError):
    """Raised when provider rate limit is hit."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, status_code=429)


class SafetyPolicyViolationError(FoundationModelError):
    """Raised when Azure AI Content Safety flags harmful input/output."""
    def __init__(self, message: str, flagged_categories: list):
        super().__init__(message, provider="azure_content_safety", status_code=400)
        self.flagged_categories = flagged_categories


class ProviderUnavailableError(FoundationModelError):
    """Raised when all primary and fallback providers fail."""
    def __init__(self, message: str):
        super().__init__(message, provider="all", status_code=503)
