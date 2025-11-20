"""Generated evaluation code for: Late pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrazole ring formation.
    Rewards routes where the specified pyrazole substructure is formed
    at or after the target step position from the end of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config["timing"]
        self.step_position = config["step_position"]
        self.pyrazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        # For late-stage formation, lower depth fractions are better
        # (closer to the target molecule)
        if self.timing == "late":
            return max(0, 1 - x) * 10
        else:
            # For early formation, higher depth fractions are better
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms the pyrazole ring by comparing
        reactants and products for the presence of the pyrazole pattern.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse products and reactants
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            # Check if pyrazole pattern is present in products but not in reactants
            pyrazole_in_products = any(
                mol and mol.HasSubstructMatch(self.pyrazole_pattern) 
                for mol in products
            )
            
            pyrazole_in_reactants = any(
                mol and mol.HasSubstructMatch(self.pyrazole_pattern) 
                for mol in reactants
            )
            
            # Ring formation occurs if pyrazole is in products but not in reactants
            return pyrazole_in_products and not pyrazole_in_reactants
            
        except Exception:
            return False
