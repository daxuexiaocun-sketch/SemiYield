"""Compatibility alias for shared evaluate utilities."""

import sys

from semiyield.common import evaluate as _implementation

sys.modules[__name__] = _implementation
