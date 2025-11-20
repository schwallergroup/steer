"""Generated evaluation code for: Late stage cyclopropane formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStage_CyclopropaneFormation(BaseScoring):
    """
    Evaluates routes based on late-stage cyclopropane formation.
    Rewards routes where cyclopropane rings are formed in the final synthetic steps.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CC1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        else:
            # Late-stage formation is better, so lower depth fractions get higher scores
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves cyclopropane formation.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            prod = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not prod or not all(reactants):
                return False
            
            # Count cyclopropane rings in product
            prod_rings = len(prod.GetSubstructMatches(self.ring_pattern))
            
            # Count cyclopropane rings in all reactants combined
            reactant_rings = sum(len(r.GetSubstructMatches(self.ring_pattern)) for r in reactants)
            
            # Check for ring formation (more rings in product than reactants)
            if self.direction == "formation":
                return prod_rings > reactant_rings
            elif self.direction == "breaking":
                return prod_rings < reactant_rings
            else:
                return prod_rings != reactant_rings
                
        except Exception:
            return False
