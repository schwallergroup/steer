"""Generated evaluation code for: Late pyrimidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrimidineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrimidine ring formation.
    Checks at what depth a pyrimidine ring is formed during the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Higher score for later formation (closer to 1.0)
        elif self.timing == "early":
            return x  # Higher score for earlier formation (closer to 0.0)
        else:
            return 0.5  # Neutral if timing not specified
    
    def hit_condition(self, d):
        """Check if pyrimidine ring formation occurs in this reaction step."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            # Product side (left of >>)
            product = Chem.MolFromSmiles(rxn_parts[0])
            if not product:
                return False
                
            # Reactants side (right of >>)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if not reactants:
                return False
            
            # Check if product contains pyrimidine ring
            product_has_ring = product.HasSubstructMatch(self.ring_pattern)
            
            # Check if any reactant contains pyrimidine ring
            reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            
            if self.direction == "formation":
                # Ring formation: product has ring but reactants don't
                return product_has_ring and not reactants_have_ring
            elif self.direction == "breaking":
                # Ring breaking: reactants have ring but product doesn't
                return not product_has_ring and reactants_have_ring
            else:
                return False
                
        except Exception:
            return False
