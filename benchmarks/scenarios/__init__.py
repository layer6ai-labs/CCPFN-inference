from ...src.ccpfn.benchmarks.scenarios.base import Scenario
from ...src.ccpfn.benchmarks.scenarios.admit import ADMIT

SCENARIOS: dict[str, type[Scenario]] = {
    "admit": ADMIT,
}

__all__ = [
    "Scenario",
    "SCENARIOS",
    "ADMIT",
]