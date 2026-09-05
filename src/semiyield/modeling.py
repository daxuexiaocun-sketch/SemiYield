"""Compatibility alias for shared modeling utilities."""

import sys

from semiyield.common import modeling as _implementation

sys.modules[__name__] = _implementation
