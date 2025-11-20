"""Generated evaluation code for: Late oxadiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OxadiazoleRingFormation(BaseScoring):
    """
    Evaluates routes based on when oxadiazole ring formation occurs.
    Rewards late-stage formation of 1,2,4-oxadiazole rings.
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
            elif self.timing == "early":
                return x  # Earlier formation is better
            else:
                return 0.5  # Neutral if timing not specified
    
    def hit_condition(self, d):
        """Check if oxadiazole ring formation occurs in this reaction step"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if self.direction == "formation":
                # Check if product has oxadiazole but reactants don't
                product_has_ring = products.HasSubstructMatch(self.ring_pattern)
                reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants if r is not None)
                
                return product_has_ring and not reactants_have_ring
                
            elif self.direction == "breaking":
                # Check if reactants have oxadiazole but product doesn't
                product_has_ring = products.HasSubstructMatch(self.ring_pattern)
                reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants if r is not None)
                
                return not product_has_ring and reactants_have_ring
                
        except Exception:
            return False
        
        return False
