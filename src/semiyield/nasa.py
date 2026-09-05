"""Compatibility alias for :mod:`semiyield.reliability.nasa`."""

import sys

from semiyield.reliability import nasa as _implementation

sys.modules[__name__] = _implementation
