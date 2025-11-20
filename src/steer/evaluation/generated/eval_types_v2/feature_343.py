"""Generated evaluation code for: Early pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrazoleFormation(BaseScoring):
    """
    Evaluates whether pyrazole ring formation occurs early in the synthesis route.
    Uses depth-based scoring where earlier formation gets higher scores.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            return 1 - x  # Earlier formation gets higher score
        else:  # late
            return x  # Later formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Checks if pyrazole ring formation occurs in this reaction step.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0].split(".")
            product_smiles = rxn[1]
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles]
            product = Chem.MolFromSmiles(product_smiles)
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains pyrazole ring
            product_has_pyrazole = product.HasSubstructMatch(self.ring_pattern)
            
            # Check if any reactant contains pyrazole ring
            reactants_have_pyrazole = any(
                mol.HasSubstructMatch(self.ring_pattern) for mol in reactants
            )
            
            if self.direction == "formation":
                # Ring formation: product has pyrazole but reactants don't
                return product_has_pyrazole and not reactants_have_pyrazole
            else:  # breaking
                # Ring breaking: reactants have pyrazole but product doesn't
                return reactants_have_pyrazole and not product_has_pyrazole
                
        except Exception:
            return False
