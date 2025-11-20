"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagePyrazoleFormation(BaseScoring):
    """
    Evaluates whether pyrazole ring formation occurs at late stages of synthesis.
    Rewards routes where pyrazole rings are formed closer to the final steps.
    """
    
    def __init__(self, config: Dict):
        self.pyrazole_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.pyrazole_pattern = Chem.MolFromSmarts(self.pyrazole_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrazole formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (closer to 1.0)
            else:
                return x  # Earlier formation is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves pyrazole ring formation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count pyrazole rings in product vs reactants
            product_pyrazole_count = len(product.GetSubstructMatches(self.pyrazole_pattern))
            reactants_pyrazole_count = sum(
                len(r.GetSubstructMatches(self.pyrazole_pattern)) for r in reactants
            )
            
            # Check for ring formation (more pyrazole rings in product than reactants)
            if self.direction == "formation":
                return product_pyrazole_count > reactants_pyrazole_count
            elif self.direction == "breaking":
                return product_pyrazole_count < reactants_pyrazole_count
            
            return False
            
        except Exception:
            return False
