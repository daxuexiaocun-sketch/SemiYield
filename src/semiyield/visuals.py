"""Compatibility alias for shared visuals utilities."""

import sys

from semiyield.common import visuals as _implementation

sys.modules[__name__] = _implementation
