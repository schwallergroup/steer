"""Generated evaluation code for: Late stage pteridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage formation of a specific ring system.
    Checks if a pteridine ring (or specified ring pattern) is formed after a given
    depth threshold in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config["timing"]
        self.step_threshold = config["step_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # For late-stage formation, reward formations that occur after threshold
            if x >= self.step_threshold:
                return 10  # Perfect score for very late formation
            else:
                # Penalty for early formation, scaled by how early
                return max(0, 10 * (x / self.step_threshold))
        elif self.timing == "early":
            # For early-stage formation, reward formations before threshold
            if x <= (1 - self.step_threshold):
                return 10
            else:
                return max(0, 10 * ((1 - x) / self.step_threshold))
        else:
            # Default: prefer later formation
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction forms the target ring system.
        Ring formation is detected by checking if the ring pattern is present
        in products but not in all reactants.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        if ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse products
        products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
        products = [p for p in products if p is not None]
        
        # Parse reactants  
        reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        reactants = [r for r in reactants if r is not None]
        
        if not products or not reactants:
            return False
            
        # Check if ring pattern is present in any product
        ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
        
        if not ring_in_products:
            return False
            
        # Check if ring pattern is already present in all reactants
        # If it's in all reactants, this isn't a ring formation reaction
        ring_in_reactants = all(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
        
        # Ring formation occurs if ring is in products but not in all reactants
        return ring_in_products and not ring_in_reactants
