"""Compatibility alias for shared preprocessing utilities."""

import sys

from semiyield.common import preprocessing as _implementation

sys.modules[__name__] = _implementation
