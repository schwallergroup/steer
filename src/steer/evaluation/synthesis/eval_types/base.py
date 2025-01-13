"""Base class for scoring synthetic routes.
For a given synthetic route, define a score - this is query dependent."""

from typing import Callable, List, Tuple

from steer.logger import setup_logger

logger = setup_logger(__name__)


class BaseScoring:
    """Find out at which depth of the tree a condition is met."""

    def __call__(self, data) -> Tuple[List[float], List[float]]:  # type: ignore
        """Provide a score based on the depth at which the condition is met."""
        cond_depth = [self.condition_depth(d["children"][0]) + 1 for d in data]
        raw_sc = [x / self.route_length(d) for x, d in zip(cond_depth, data)]
        score = [10 * self.route_scoring(x) for x in raw_sc]
        lmscore = [d["lmdata"]["routescore"] for d in data]
        return score, lmscore

    def hit_condition(self, d):  # type: ignore
        "Define hit condition: define what we are looking for."
        pass

    def route_scoring(self, x) -> float:  # type: ignore
        """Define scoring function.
        x: depth at which condition is met in route / length of route."""
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
            """Depth first search to find all paths."""
            if "children" in d:
                if d["type"] == "reaction":
                    curr_path.append(d["smiles"])
                for c in d["children"]:
                    yield from dfs(c, curr_path)
            else:
                yield curr_path

        total_depth = [len(p) for p in dfs(data, [])]
        return max(total_depth)


if __name__ == "__main__":
    import json

    with open("../../../data/aizynth_output.json", "r") as f:
        data = json.load(f)

    bs = BaseScoring()
    for d in data:
        print(bs.route_length(d))
