"""Generated evaluation code for: Late oxazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage oxazole ring formation.
    Rewards routes where oxazole rings are formed in the final synthetic steps,
    typically via methods like Hantzsch synthesis from α-bromoketone and acetamide.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ocnc1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better (lower depth fraction gets higher score)
            # Scale from 0-10 where formation at depth 0 gets score 10
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms an oxazole ring"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains oxazole ring
            oxazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not product.HasSubstructMatch(oxazole_pattern):
                return False
            
            # Check if oxazole ring is formed (not present in reactants)
            for reactant in reactants:
                if reactant.HasSubstructMatch(oxazole_pattern):
                    return False  # Ring already exists in reactants
            
            return True  # Ring formed in this step
            
        except Exception:
            return False
