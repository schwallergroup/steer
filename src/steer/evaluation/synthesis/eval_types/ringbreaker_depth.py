"""RingBreakerDepth scoring function"""

from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import BaseScoring


class RingBreakDepth(BaseScoring):
    def __init__(self, config: Dict):  # type: ignore
        """Config defines the scoring function based on target_depth"""
        self.condition_type = config["target_depth"]["type"]
        self.target_depth = config["target_depth"]["value"]

    def route_scoring(self, x) -> float:  # type: ignore
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
        """We're looking specifically for ringbreaking(forming) reactions."""
        rxn = d["metadata"]["mapped_reaction_smiles"]
        return self.is_ring_forming_reaction_with_mapping(rxn)

    def is_ring_forming_reaction_with_mapping(self, rxn):
        prds, rcts = rxn.split(">>")
        reactant = Chem.MolFromSmiles(rcts)
        product = Chem.MolFromSmiles(prds)

        # Build a mapping: mapnum -> reactant atom idx
        reactant_mapnums = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in reactant.GetAtoms()
            if atom.GetAtomMapNum() != 0
        }

        product_ri = product.GetRingInfo()
        reactant_ri = reactant.GetRingInfo()
        # For each atom in product, check mapping number
        for atom in product.GetAtoms():
            mapnum = atom.GetAtomMapNum()
            if mapnum == 0:
                continue
            product_atom_idx = atom.GetIdx()
            in_ring_product = np.any(
                [
                    product_ri.IsAtomInRingOfSize(product_atom_idx, i)
                    for i in range(2, 10)
                ]
            )
            # Find corresponding atom in reactant
            if mapnum in reactant_mapnums:
                reactant_atom_idx = reactant_mapnums[mapnum]
                in_ring_reactant = np.any(
                    [
                        reactant_ri.IsAtomInRingOfSize(reactant_atom_idx, i)
                        for i in range(2, 10)
                    ]
                )
                # If atom is in ring in product but not in reactant, ring formed
                if in_ring_product and not in_ring_reactant:
                    return True
        return False


if __name__ == "__main__":
    import json

    with open(
        "../../../data/2025-01-08_204803/280b79ef56e06a8af1a7d6b72c52148d.json",
        "r",
    ) as f:
        data = json.load(f)

    bs = RingBreakDepth(config={"target_depth": {"type": "diff", "value": 1}})
    print(bs(data))
