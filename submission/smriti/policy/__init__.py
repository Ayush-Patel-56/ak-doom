"""Policies. Phase 1 only has the random baseline; the trained brain arrives in Phase 2."""

from .random_policy import RandomPolicy

__all__ = ["RandomPolicy"]
