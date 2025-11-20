"""Generated evaluation code for: Late pyrimidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrimidineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrimidine ring formation.
    Rewards routes where pyrimidine rings are formed later in the synthesis,
    indicating strategic bond construction that builds complexity late.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (closer to 1.0)
            else:
                return x  # Earlier formation is better
    
    def hit_condition(self, d) -> bool:
        """Check if pyrimidine ring formation occurs in this reaction step."""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count pyrimidine rings in product
            product_rings = len(product.GetSubstructMatches(self.ring_pattern))
            
            # Count pyrimidine rings in all reactants combined
            reactant_rings = sum(len(r.GetSubstructMatches(self.ring_pattern)) for r in reactants)
            
            if self.direction == "formation":
                # Ring formation: more rings in product than reactants
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                # Ring breaking: fewer rings in product than reactants
                return product_rings < reactant_rings
            else:
                # Any change in ring count
                return product_rings != reactant_rings
                
        except (KeyError, AttributeError, ValueError):
            return False
