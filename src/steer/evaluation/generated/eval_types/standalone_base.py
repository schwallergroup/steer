"""Standalone base classes for evaluation (no external dependencies).

This is a copy of the base scoring classes from steer.evaluation.synthesis.eval_types
but without any imports from the main steer package to avoid dependency issues.
"""

from typing import List, Tuple


class BaseScoring:
    """Find out at which depth of the tree a condition is met."""

    def __call__(self, data) -> Tuple[List[float], List[float]]:
        """Provide a score based on the depth at which the condition is met."""
        cond_depth = [self.condition_depth(d["children"][0]) + 1 for d in data]
        raw_sc = [x / self.route_length(d) for x, d in zip(cond_depth, data)]
        score = [10 * self.route_scoring(x) for x in raw_sc]
        lmscore = [d["lmdata"]["routescore"] for d in data]
        return score, lmscore

    def hit_condition(self, d):
        "Define hit condition: define what we are looking for."
        pass

    def route_scoring(self, x) -> float:
        """Define scoring function.
        x: depth at which condition is met in route / length of route."""
        pass

    def condition_depth(self, d, i=0):
        """bfs search for reaction that matches hit condition."""
        negative = -2
        if self.hit_condition(d):
            return i
        if "children" in d:
            for c in d["children"]:
                if "children" in c:
                    a = self.condition_depth(c["children"][0], i + 1)
                    if a != negative:
                        return a
        return negative

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


class MultiRxnCondBase:
    """Base class for checking multiple reactions in tree."""

    def __call__(self, data):
        """For all routes found (and scored) in the data, find the depth at which the hit condition is met, and plot."""

        score, lengths, lmscore = [], [], []
        for d in data:
            cond, length = self.condition_depth(d["children"][0])
            score.append(10 if cond else 0)
            lengths.append(length)
            lmscore.append(d["lmdata"]["routescore"])
        return score, lmscore

    def detect_specific_break(self, rxn, pattern):
        """Detect if a specific bond is broken in a reaction."""
        from rdkit import Chem
        p = Chem.MolFromSmarts(pattern)
        prod = Chem.MolFromSmiles(rxn.split(">>")[0])
        reac = Chem.MolFromSmiles(rxn.split(">>")[1])
        return prod.HasSubstructMatch(p) and not reac.HasSubstructMatch(p)

    def get_rxns(self, d):
        """Extract all the reactions from tree."""

        def _extract_reactions(d):
            if "metadata" in d:
                yield d["metadata"]["mapped_reaction_smiles"]
                for c in d["children"]:
                    if "children" in c:
                        for r in _extract_reactions(c["children"][0]):
                            yield r

        reactions = list(_extract_reactions(d))
        return reactions
