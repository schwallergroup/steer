"""Generated evaluation code for: Late stage lactam ring closure"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageLactamClosure(BaseScoring):
    """
    Evaluates whether a late-stage lactam ring formation occurs in the synthesis route.
    Detects formation of six-membered lactam rings and rewards later occurrence in the route.
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
            # For late-stage preference, higher depth fraction is better
            if self.timing == "late":
                return x  # Score increases with depth (0-1 range)
            else:
                return 1 - x  # Early stage would prefer lower depth
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node involves lactam ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains the lactam ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant already contains the lactam ring
            # If formation is desired, reactants should NOT have the ring
            if self.direction == "formation":
                for reactant in reactants:
                    if reactant.HasSubstructMatch(self.ring_pattern):
                        return False  # Ring already exists, not a formation
                return True  # Ring in product but not in reactants = formation
            
            # If breaking is desired (though not in this case)
            elif self.direction == "breaking":
                for reactant in reactants:
                    if reactant.HasSubstructMatch(self.ring_pattern):
                        return True  # Ring in reactants but not in product = breaking
                return False
            
            return False
            
        except (KeyError, AttributeError, ValueError):
            return False
