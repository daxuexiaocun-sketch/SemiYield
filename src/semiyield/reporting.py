"""Compatibility alias for shared reporting utilities."""

import sys

from semiyield.common import reporting as _implementation

sys.modules[__name__] = _implementation
