"""Generated evaluation code for: Late isoxazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIsoxazoleFormation(BaseScoring):
    """
    Evaluates whether isoxazole ring formation occurs late in the synthesis route.
    Checks for the formation of isoxazole rings ([#6]1[#6][#7][#8][#6]1) and scores
    based on how late in the route this formation occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late timing, later formation (higher x) gets better score
            if self.timing == "late":
                return x  # x is already 0-1, scale to 0-10 will happen elsewhere
            else:
                return 1 - x  # For early timing, earlier formation is better
    
    def hit_condition(self, d):
        """
        Checks if isoxazole ring formation occurs in this reaction step.
        Returns True if the reaction forms an isoxazole ring.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[1].split(".")]
            
            # Count isoxazole rings in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                               for mol in reactants if mol is not None)
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                              for mol in products if mol is not None)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_rings > reactant_rings
            elif self.direction == "break":
                # Ring breaking: fewer rings in products than reactants
                return reactant_rings > product_rings
            else:
                return False
                
        except (KeyError, AttributeError, ValueError):
            return False
