"""Generated evaluation code for: Late stage cyclopropanation as final step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates if a specific ring formation occurs as the final step in the synthesis.
    Checks for cyclopropane ring formation (or other specified rings) in the last reaction.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        elif self.timing == "final":
            # For final step requirement, only depth 0 (final step) gets full score
            if x == 0:
                return 10
            else:
                return max(0, 10 - x * 5)  # Penalty for earlier occurrence
        else:
            # For late-stage preference, later is better
            return max(0, 10 - x * 2)
    
    def hit_condition(self, d) -> bool:
        """Check if the reaction involves formation of the target ring structure."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not product or not reactants:
                return False
            
            # Count ring matches in product
            product_rings = len(product.GetSubstructMatches(self.ring_pattern))
            
            # Count ring matches in all reactants combined
            reactant_rings = sum(len(r.GetSubstructMatches(self.ring_pattern)) 
                               for r in reactants if r is not None)
            
            # Check if ring formation occurred (more rings in product than reactants)
            if self.direction == "formation":
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                return product_rings < reactant_rings
            
        except Exception:
            return False
        
        return False
