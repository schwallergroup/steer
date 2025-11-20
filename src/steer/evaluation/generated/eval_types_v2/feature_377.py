"""Generated evaluation code for: Late stage cyclopropane formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclopropaneFormation(BaseScoring):
    """
    Evaluates if cyclopropane ring formation occurs in the later stages of synthesis.
    Uses stage_threshold to define what constitutes "late stage" (default 0.7 = last 30% of route).
    Returns higher scores for later cyclopropane formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.7)
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation not found
        
        # x is the depth fraction where cyclopropane formation occurs
        # Higher scores for later formation (closer to 1.0)
        if x >= self.stage_threshold:
            # Late stage formation - reward based on how late
            return 8 + 2 * (x - self.stage_threshold) / (1 - self.stage_threshold)
        else:
            # Early stage formation - penalize based on how early
            return 4 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a cyclopropane ring.
        Looks for cyclopropane present in product but not in all reactants.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if product contains cyclopropane
            product_has_cyclopropane = product.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_cyclopropane:
                return False
                
            # Check if cyclopropane is newly formed (not present in all reactants)
            reactant_cyclopropanes = sum(1 for r in reactants if r.HasSubstructMatch(self.ring_pattern))
            product_cyclopropanes = len(product.GetSubstructMatches(self.ring_pattern))
            
            # Cyclopropane formation if product has more cyclopropanes than sum of reactants
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
