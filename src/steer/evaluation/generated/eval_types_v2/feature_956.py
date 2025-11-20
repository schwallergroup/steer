"""Generated evaluation code for: Late stage triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTriazoleFormation(BaseScoring):
    """
    Evaluates routes for late-stage triazole ring formation.
    Checks if a 1,2,4-triazole ring is formed after a specified stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"] 
        self.stage_threshold = config["parameters"]["stage_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                # Reward later formation, penalize earlier formation
                if x >= self.stage_threshold:
                    return 10 * (1 - (1 - x) / (1 - self.stage_threshold))  # Scale 0.8-1.0 to 0-10
                else:
                    return 0  # Too early, no reward
            else:
                return 10 * (1 - x)  # General late-stage preference
    
    def hit_condition(self, d) -> bool:
        """Check if triazole ring is formed in this reaction step"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r)]
            
            if not product or not reactants:
                return False
            
            # Check if product contains triazole ring
            product_has_triazole = product.HasSubstructMatch(self.ring_pattern)
            
            # Check if any reactant already has triazole ring
            reactant_has_triazole = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            
            # Ring formation occurs if product has triazole but reactants don't
            return product_has_triazole and not reactant_has_triazole
            
        except Exception:
            return False
