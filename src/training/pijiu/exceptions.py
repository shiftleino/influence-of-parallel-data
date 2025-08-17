__all__ = [
    "PijiuError",
    "PijiuConfigurationError",
    "PijiuCliError",
    "PijiunvironmentError",
    "PijiuNetworkError",
    "PijiuCheckpointError",
]


class PijiuError(Exception):
    """
    Base class for all custom Pijiu exceptions.
    """


class PijiuConfigurationError(PijiuError):
    """
    An error with a configuration file.
    """


class PijiuCliError(PijiuError):
    """
    An error from incorrect CLI usage.
    """


class PijiuEnvironmentError(PijiuError):
    """
    An error from incorrect environment variables.
    """


class PijiuNetworkError(PijiuError):
    """
    An error with a network request.
    """


class PijiuCheckpointError(PijiuError):
    """
    An error occurred reading or writing from a checkpoint.
    """


class PijiuThreadError(Exception):
    """
    Raised when a thread fails.
    """
