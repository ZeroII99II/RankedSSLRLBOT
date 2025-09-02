from dataclasses import dataclass


@dataclass
class SSLStateSetter:  # minimal placeholder for tests
    def reset(self, state, rng):  # pragma: no cover - behaviour not needed in tests
        return state


from .scenarios import SCENARIOS

__all__ = ["SSLStateSetter", "SCENARIOS"]

