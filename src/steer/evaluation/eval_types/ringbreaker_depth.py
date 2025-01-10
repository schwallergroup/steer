"""RingBreakerDepth scoring function"""

from typing import Dict

from .base import BaseScoring


class RingBreakDepth(BaseScoring):
    def __init__(self, config: Dict):
        """Config defines the scoring function based on target_depth"""
        if config["target_depth"]["type"] == "bool":
            self.target_depth = lambda x: x == config["target_depth"]["value"]
        elif config["target_depth"]["type"] == "diff":
            self.target_depth = lambda x: abs(
                x - config["target_depth"]["value"]
            )

    def hit_condition(self, d):
        return d.get("metadata", {}).get("policy_name") == "ringbreaker"

    def __call__(self, data):
        return self.where_condition_met(data, self.target_depth)


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