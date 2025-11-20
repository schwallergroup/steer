"""Generated evaluation code for: Early stage triazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoloneFormationDepth(BaseScoring):
    """
    Evaluates the depth at which triazolone ring formation occurs in a synthesis route.
    Rewards early-stage formation of triazolone heterocycles via intramolecular cyclization.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Earlier formation is better (higher score)
            else:
                return x  # Later formation is better (lower depth fraction gives higher score)
    
    def hit_condition(self, d):
        """
        Checks if triazolone ring formation occurs at this reaction step.
        """
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        product = Chem.MolFromSmiles(rxn[0])
        
        if not product or not all(reactants):
            return False
        
        # Check if product contains triazolone ring
        product_has_ring = product.HasSubstructMatch(self.ring_pattern)
        
        # Check if any reactant contains triazolone ring
        reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
        
        # Ring formation: product has ring but reactants don't
        if self.direction == "formation":
            return product_has_ring and not reactant_has_ring
        # Ring breaking: reactants have ring but product doesn't
        elif self.direction == "breaking":
            return not product_has_ring and reactant_has_ring
        
        return False
