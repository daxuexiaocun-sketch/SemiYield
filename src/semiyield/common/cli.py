"""Shared CLI error presentation."""

from functools import wraps

import typer


def friendly_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, FileNotFoundError, ImportError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    return wrapped
