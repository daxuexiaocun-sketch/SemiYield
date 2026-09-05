"""Device lifetime analysis; legacy imports remain supported."""

from .models import *  # noqa: F403


def __getattr__(name):
    # Resolve legacy private helpers as well as the public exports above.
    from . import models

    return getattr(models, name)
