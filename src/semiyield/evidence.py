"""Compatibility alias for shared evidence utilities."""

import sys

from semiyield.common import evidence as _implementation

sys.modules[__name__] = _implementation
