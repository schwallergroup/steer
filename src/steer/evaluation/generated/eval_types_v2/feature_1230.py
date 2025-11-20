"""Generated evaluation code for: Late stage thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThiazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation.
    
    Checks if a thiazole ring (c1scnc1) is formed after the specified stage cutoff.
    Returns higher scores for later formation, penalizes early formation or absence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.stage_cutoff = config["parameters"]["stage_cutoff"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.direction == "late":
            if x >= self.stage_cutoff:
                # Late formation - higher score for later stages
                return 8 + (x - self.stage_cutoff) * 2 / (1 - self.stage_cutoff)
            else:
                # Too early - penalize
                return 2 * (x / self.stage_cutoff)
        else:
            # Early formation preferred
            if x <= self.stage_cutoff:
                return 8 + (self.stage_cutoff - x) * 2 / self.stage_cutoff
            else:
                return 2 * ((1 - x) / (1 - self.stage_cutoff))
    
    def hit_condition(self, d) -> bool:
        """Check if thiazole ring is formed in this reaction step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".") if r.strip()]
            
            if not product or not reactants:
                return False
            
            # Check if product contains thiazole ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant already contains the complete thiazole ring
            for reactant in reactants:
                if reactant and reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already exists, not formed in this step
            
            return True  # Ring is formed in this step
            
        except Exception:
            return False
