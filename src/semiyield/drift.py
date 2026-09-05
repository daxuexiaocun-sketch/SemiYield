"""Compatibility alias for shared drift utilities."""

import sys

from semiyield.common import drift as _implementation

sys.modules[__name__] = _implementation
