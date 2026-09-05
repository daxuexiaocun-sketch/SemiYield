"""Compatibility alias for shared predict utilities."""

import sys

from semiyield.common import predict as _implementation

sys.modules[__name__] = _implementation
