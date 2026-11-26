"""Policies. RandomPolicy is the fallback; BrainPolicy runs the trained SmritiNet (needs torch)."""

from .random_policy import RandomPolicy

__all__ = ["RandomPolicy"]
