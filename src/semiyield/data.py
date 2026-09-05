"""Compatibility alias for :mod:`semiyield.yield_risk.data`."""

import sys

from semiyield.yield_risk import data as _implementation

sys.modules[__name__] = _implementation
