from .base import Scenario
from .admit import ADMIT

SCENARIOS: dict[str, type[Scenario]] = {
    "admit": ADMIT,
}

__all__ = [
    "Scenario",
    "SCENARIOS",
    "ADMIT",
]