class ModelzError(Exception):
    """Base class for expected modelz failures (as opposed to bugs)."""


class ConfigError(ModelzError):
    """Raised when required configuration (e.g. the API token) is missing."""


class InputValidationError(ModelzError):
    """Raised when a user-supplied file isn't a usable image/video."""


class StageFailedError(ModelzError):
    """Raised when a pipeline stage (Replicate call or local processing) fails."""
