"""Compatibility alias for :mod:`semiyield.yield_risk.benchmark`."""

import sys

from semiyield.yield_risk import benchmark as _implementation

sys.modules[__name__] = _implementation
