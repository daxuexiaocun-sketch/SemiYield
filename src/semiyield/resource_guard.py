"""Compatibility alias for shared resource_guard utilities."""

import sys

from semiyield.common import resource_guard as _implementation

sys.modules[__name__] = _implementation
