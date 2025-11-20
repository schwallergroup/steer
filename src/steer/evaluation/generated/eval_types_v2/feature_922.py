"""Generated evaluation code for: Late oxadiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoleRingFormation(BaseScoring):
    """
    Evaluates whether oxadiazole ring formation occurs late in the synthesis route.
    
    Checks for the formation of 1,2,4-oxadiazole rings ([#6]1[#7][#8][#6][#7]1)
    and scores based on how late in the route this ring formation occurs.
    Late-stage formation is preferred (higher score).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score where late formation gets higher score.
        
        Args:
            x: Depth fraction where ring formation occurs (-1 if not found)
            
        Returns:
            Score from 0-1 where 1 is best (latest formation)
        """
        if x < 0:
            return 0  # Ring formation not found
        
        if self.timing == "late":
            # Higher depth fraction (closer to 1) is better for late timing
            return x
        else:
            # For early timing, lower depth fraction is better
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves oxadiazole ring formation.
        
        Args:
            d: Reaction node dictionary containing metadata
            
        Returns:
            True if oxadiazole ring is formed in this step
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains oxadiazole ring
            product_has_ring = product.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_ring:
                return False
            
            if self.direction == "formation":
                # Ring formation: product has ring but reactants don't
                reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
                return not reactants_have_ring
            else:
                # Ring breaking: reactants have ring but product doesn't
                reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
                return reactants_have_ring
                
        except Exception:
            return False
