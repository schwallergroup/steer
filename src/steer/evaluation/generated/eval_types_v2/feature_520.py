"""Generated evaluation code for: Late stage piperazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when a specific ring is formed.
    Rewards late-stage ring formation at the specified depth.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.target_depth = config["parameters"]["formation_depth"]
        self.timing = config["parameters"]["timing"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Reward formation at target depth, penalize earlier formation
            if x <= self.target_depth / 10.0:  # Convert depth to fraction
                return 10  # Perfect score for late formation
            else:
                return max(0, 10 - (x * 10 - self.target_depth) * 2)  # Penalty for early formation
        else:
            # For early timing, reward earlier formation
            return 10 - abs(x * 10 - self.target_depth)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms the target ring structure"""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Parse product and reactants
        product = Chem.MolFromSmiles(rxn_parts[0])
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
        
        if not product or not all(reactants):
            return False
        
        # Check if product contains the target ring
        product_has_ring = product.HasSubstructMatch(self.ring_pattern)
        
        # Check if any reactant already contains the complete ring
        reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
        
        # Ring formation occurs if product has ring but reactants don't
        return product_has_ring and not reactants_have_ring
