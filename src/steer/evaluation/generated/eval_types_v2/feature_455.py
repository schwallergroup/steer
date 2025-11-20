"""Generated evaluation code for: Late stage purine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates late-stage purine ring formation in synthesis routes.
    Checks if purine ring system is formed via intramolecular cyclization
    at or before the specified depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # For late-stage formation, penalize early formation
            if x <= self.depth_threshold / 10.0:  # Convert depth to fraction
                return 10  # Excellent - formed at target depth or later
            else:
                return max(0, 10 - (x - self.depth_threshold / 10.0) * 50)
        else:
            # For early-stage formation, penalize late formation
            return max(0, 10 - x * 10)
    
    def hit_condition(self, d):
        """Check if this reaction involves purine ring formation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains purine ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant already contains the purine ring
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already present, not formation
            
            # If product has purine but reactants don't, this is ring formation
            return True
            
        except:
            return False
