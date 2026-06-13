"""Shared exception types."""


class PipelineError(Exception):
    """Expected, user-readable pipeline failure (surfaced in the job's error field)."""


class NotReadyError(PipelineError):
    """A pipeline stage was requested before its prerequisites exist."""
