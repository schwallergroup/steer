"""Generated evaluation code for: Late stage triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTriazoleFormation(BaseScoring):
    """
    Evaluates whether triazole ring formation occurs late in the synthesis route.
    Detects formation of 1,2,4-triazole rings from precursors and rewards
    late-stage formation over early-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.triazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Triazole formation doesn't occur
        else:
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score
            else:
                return x  # Earlier formation gets higher score
    
    def hit_condition(self, d):
        """
        Detects triazole ring formation by checking if:
        1. Product contains the triazole pattern
        2. At least one reactant lacks the complete triazole pattern
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains triazole pattern
            if not product.HasSubstructMatch(self.triazole_pattern):
                return False
            
            # Check if triazole is being formed (not already present in all reactants)
            triazole_in_reactants = [r.HasSubstructMatch(self.triazole_pattern) for r in reactants]
            
            if self.direction == "formation":
                # Formation: product has triazole, but not all reactants do
                return not all(triazole_in_reactants)
            elif self.direction == "breaking":
                # Breaking: all reactants have triazole, product doesn't
                return all(triazole_in_reactants)
            
            return False
            
        except Exception:
            return False
