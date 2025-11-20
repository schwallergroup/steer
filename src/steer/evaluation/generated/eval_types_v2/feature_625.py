"""Generated evaluation code for: Late oxazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxazoleRingFormation(BaseScoring):
    """
    Evaluates routes for late-stage oxazole ring formation at a specific depth.
    Checks for the formation of oxazole rings (c1ocnc1) through ring-forming reactions
    like Robinson-Gabriel type cyclocondensations.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.target_depth = config["parameters"]["formation_depth"]
        self.timing = config["parameters"]["timing"]
        self.oxazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Late-stage formation is preferred (lower depth values are better)
            if x <= self.target_depth / 10.0:  # Convert depth to fraction
                return 10  # Perfect score for formation at target depth or later
            else:
                return max(0, 10 - (x - self.target_depth / 10.0) * 20)
        else:
            # Early-stage formation
            return max(0, 10 - abs(x - self.target_depth / 10.0) * 20)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms an oxazole ring by comparing
        reactants and products for oxazole substructure presence.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse products
            product = Chem.MolFromSmiles(products_smiles)
            if not product:
                return False
                
            # Check if product contains oxazole
            has_oxazole_in_product = product.HasSubstructMatch(self.oxazole_pattern)
            
            if not has_oxazole_in_product:
                return False
            
            # Parse reactants
            reactant_smiles_list = reactants_smiles.split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles_list]
            reactants = [r for r in reactants if r is not None]
            
            # Check if any reactant already contains the oxazole ring
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.oxazole_pattern):
                    return False  # Ring already present, not a formation reaction
            
            # If product has oxazole but no reactant does, this is ring formation
            return True
            
        except (KeyError, ValueError, AttributeError):
            return False
