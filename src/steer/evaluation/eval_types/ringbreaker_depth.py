"""RingBreakerDepth scoring function"""

from typing import Dict

from .base import BaseScoring


class RingBreakDepth(BaseScoring):
    def __init__(self, config: Dict):
        """Config defines the scoring function based on target_depth"""
        self.condition_type = config["target_depth"]["type"]
        self.target_depth = config["target_depth"]["value"]

    def route_scoring(self, x) -> float:
        """x: depth at which condition is met in route / length of route."""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
            # Else idk
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)

    def hit_condition(self, d):
        return d.get("metadata", {}).get("policy_name") == "ringbreaker"


if __name__ == "__main__":
    import json

    with open("../../../data/2025-01-08_204803/280b79ef56e06a8af1a7d6b72c52148d.json", "r") as f:
        data = json.load(f)

    bs = RingBreakDepth(
        config={
            "target_depth": {
                "type": "diff",
                "value": 1
            }
        }
    )
    print(bs(data))