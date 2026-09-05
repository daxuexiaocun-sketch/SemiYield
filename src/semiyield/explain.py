"""Compatibility alias for shared explain utilities."""

import sys

from semiyield.common import explain as _implementation

sys.modules[__name__] = _implementation
