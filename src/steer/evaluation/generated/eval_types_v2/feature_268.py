"""Generated evaluation code for: Late oxadiazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoloneFormation(BaseScoring):
    """
    Evaluates whether oxadiazolone ring formation occurs in late stages of synthesis.
    
    Checks for the formation of oxadiazolone rings ([#6]1[#7][#8][#6](=O)[#7]1) and
    rewards when this occurs after the specified stage threshold (late in the route).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.stage_threshold = config["parameters"]["stage_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Reward ring formation after the threshold (later stages)
            if x >= self.stage_threshold:
                return 10.0  # Perfect score for late formation
            else:
                # Penalize early formation, scale from 0 to 5
                return 5.0 * (x / self.stage_threshold)
        else:
            # For other timing preferences, return inverse relationship
            return 10.0 * (1.0 - x)
    
    def hit_condition(self, d):
        """Check if this reaction step forms an oxadiazolone ring."""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        reactants_smiles, product_smiles = rxn_smiles.split(">>")
        
        # Parse molecules
        product = Chem.MolFromSmiles(product_smiles)
        reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        
        if not product or not all(reactants):
            return False
        
        # Check if product contains oxadiazolone ring
        product_has_ring = product.HasSubstructMatch(self.ring_pattern)
        
        if not product_has_ring:
            return False
        
        # Check if any reactant already has the oxadiazolone ring
        reactant_has_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants if r)
        
        # Ring formation occurs if product has ring but reactants don't
        return product_has_ring and not reactant_has_ring
