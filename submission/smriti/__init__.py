"""Smriti: the AK-Doom agent's brain.

Lives inside submission/ so the scorer and validator can import it (they put submission/ on
sys.path before loading agent.py). Training code imports the same package, so there is one copy.

Rule: nothing in this package may import from training/ or scripts/ (a contract test checks it).
"""

from .safety import SAFE_ACTION, SafetyGuard

__all__ = ["SAFE_ACTION", "SafetyGuard"]
__version__ = "0.1.0"
