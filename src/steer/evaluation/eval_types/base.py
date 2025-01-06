# For a given synthetic route, define a score. This is query dependent.
# For each query, we define a metric to evaluate the synthetic route.

from typing import Callable, List, Tuple

class BaseScoring:
    """Find out at which depth of the tree a condition is met."""

    def __call__(self, data) -> Tuple[List[float], List[float]]:
        """Evaluate the synthetic route."""
        pass

    def depth_score(self, d):
        return self.condition_depth(d["children"][0]) + 1

    def hit_condition(self, d):
        "Hit condition: define what we are looking for."
        pass

    def condition_depth(self, d, i=0):
        """bfs search for reaction that matches hit condition."""
        if self.hit_condition(d):
            return i
        if "children" in d:
            for c in d["children"]:
                if "children" in c:
                    a = self.condition_depth(c["children"][0], i + 1)
                    if a != -1:
                        return a
        return -2

    def route_length(self, data):
        """Find the length of the route."""
        # length = [len(d['children']) for d in data]
        # lmscore = [d['lmdata']['routescore'] for d in data]
        pass

    def where_condition_met(self, data, target_depth: Callable):
        """Provide a score based on the depth at which the condition is met."""
        depth = [self.depth_score(d) for d in data]
        lmscore = [d["lmdata"]["routescore"] for d in data]

        # For now let's say difference with target_depth is the score
        score = [target_depth(d) for d in depth]
        return score, lmscore
