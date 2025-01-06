
from .base import BaseScoring
from typing import Dict

class RingBreakDepth(BaseScoring):
    def __init__(self, config: Dict):
        """Config defines the scoring function based on target_depth"""
        if config['target_depth']["type"] == "bool":
            self.target_depth = lambda x: x == config['target_depth']["value"]
        elif config['target_depth']["type"] == "diff":
            self.target_depth = lambda x: abs(x - config['target_depth']["value"])

    def hit_condition(self, d):
        return d.get('metadata', {}).get('policy_name') == 'ringbreaker'

    def __call__(self, data):
        return self.where_condition_met(data, self.target_depth)
