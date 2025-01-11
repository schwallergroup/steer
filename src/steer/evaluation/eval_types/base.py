"""Base class for scoring synthetic routes.
For a given synthetic route, define a score - this is query dependent."""

from typing import Callable, List, Tuple
from steer.logger import setup_logger

logger = setup_logger(__name__)


class BaseScoring:
    """Find out at which depth of the tree a condition is met."""

    def __call__(self, data) -> Tuple[List[float], List[float]]:  # type: ignore
        """Evaluate the synthetic route."""
        pass

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

        def dfs(d, curr_path):
            if "children" in d:
                if d['type'] == "reaction":
                    curr_path.append(d['smiles'])
                for c in d["children"]:
                    yield from dfs(c, curr_path)
            else:
                yield curr_path

        total_depth = [len(p) for p in dfs(data, [])]
        return max(total_depth)

    def score(self, d):
        """Rule-base score of synthetic route.
        Depth at which condition is met / length of route, scaled to [0,10]"""
        cond_depth = self.condition_depth(d["children"][0]) + 1
        if cond_depth == -1:
            return -1
        else:
            return 10 * cond_depth / self.route_length(d)

    def where_condition_met(self, data, target_depth: Callable):
        """Provide a score based on the depth at which the condition is met."""
        cond_depth = [self.condition_depth(d["children"][0])+1 for d in data]

        probe = target_depth(cond_depth[0])
        if isinstance(probe, bool):
            score = [10 if target_depth(x) else 1 for x in cond_depth]
        else:
            cond_depth = [target_depth(10 * d) if d != -1 else 0 for d in cond_depth]
            score = [x / self.route_length(d) for x, d in zip(cond_depth, data)]

        lmscore = [d["lmdata"]["routescore"] for d in data]

        for i, (d,l) in enumerate(zip(cond_depth, lmscore)):
            logger.debug(f"Depth: {d}, LMScore: {l}, target_depth: {score[i]}")
        return score, lmscore

if __name__ == "__main__":
    import json

    with open("../../../data/aizynth_output.json", "r") as f:
        data = json.load(f)

    bs = BaseScoring()
    for d in data:
        print(bs.route_length(d))