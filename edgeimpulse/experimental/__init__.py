"""This module contains all our experimental features.

Be warned these signatures may change.
"""

#
# Expose all packages here so that they can be imported in this manner. Do not remove.
#
# import edgeimpulse as ei
# ei.experimental.data.<func>
#

from .api import EdgeImpulseApi

# ruff: noqa: F401
import edgeimpulse.experimental.data

# ruff: noqa: F401
import edgeimpulse.experimental.tuner

# ruff: noqa: F401
import edgeimpulse.experimental.impulse

# ruff: noqa: F401
import edgeimpulse.experimental.util

__all__ = [
    "EdgeImpulseApi",
]
