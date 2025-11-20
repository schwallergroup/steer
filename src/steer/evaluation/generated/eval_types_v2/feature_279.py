"""Generated evaluation code for: Late pyrimidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrimidineRingFormation(BaseScoring):
    """
    Evaluates routes for late-stage pyrimidine ring formation.
    Rewards routes where pyrimidine rings are formed later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (higher depth fraction = lower score)
            else:  # early timing
                return x  # Earlier formation is better (lower depth fraction = higher score)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves pyrimidine ring formation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check if product has pyrimidine ring
            if not products.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant has pyrimidine ring
            for reactant in reactants:
                if reactant and reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already exists, not formation
            
            # Ring is in product but not in reactants = ring formation
            return True
            
        except Exception:
            return False
