"""Generated evaluation code for: Late stage epoxide ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEpoxideFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage epoxide ring formation.
    Checks if an epoxide (3-membered ring with oxygen) is formed in the later stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CO1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.epoxide_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Epoxide formation doesn't happen
        else:
            # For late-stage formation, lower depth fraction is better
            # Convert to 0-10 scale where late stage gets higher score
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves epoxide formation.
        Returns True if epoxide is formed (present in product but not in reactants).
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains epoxide
            product_has_epoxide = product.HasSubstructMatch(self.epoxide_pattern)
            
            if not product_has_epoxide:
                return False
            
            # Check if any reactant already contains epoxide
            reactants_have_epoxide = any(r.HasSubstructMatch(self.epoxide_pattern) for r in reactants)
            
            # Epoxide formation: product has epoxide but reactants don't
            if self.direction == "formation":
                return product_has_epoxide and not reactants_have_epoxide
            
            return False
            
        except (KeyError, AttributeError, ValueError):
            return False
