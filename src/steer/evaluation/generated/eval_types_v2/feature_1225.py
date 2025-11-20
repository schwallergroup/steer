"""Generated evaluation code for: Late thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation.
    Detects when thiazole rings are formed and rewards later formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late timing, later formation (higher x) gets better score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 10 * x  # Later is better, max score at depth 1.0
            else:
                return 10 * (1 - x)  # Earlier is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if thiazole ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Parse reactants and product
        try:
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Count thiazole rings in product and reactants
            product_thiazoles = len(product.GetSubstructMatches(self.thiazole_pattern))
            reactant_thiazoles = sum(len(r.GetSubstructMatches(self.thiazole_pattern)) 
                                   for r in reactants)
            
            # Ring formation detected if product has more thiazole rings than reactants
            return product_thiazoles > reactant_thiazoles
            
        except:
            return False
