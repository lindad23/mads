"""Isolated curriculum baseline implementations.

This package keeps reproduced baseline logic separate from the existing MADS
runner and environment code. Training scripts should depend on the small
teacher interface exposed here instead of importing external repositories.
"""

from .teachers import (
    CPDRLTeacher,
    ProCuRLTargetTeacher,
    TaskSpace,
)
from .adapters import CPDRLAdapter, ProCuRLTargetAdapter

__all__ = [
    "CPDRLAdapter",
    "CPDRLTeacher",
    "ProCuRLTargetAdapter",
    "ProCuRLTargetTeacher",
    "TaskSpace",
]
