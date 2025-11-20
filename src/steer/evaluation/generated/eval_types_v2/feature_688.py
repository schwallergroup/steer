"""Generated evaluation code for: Late imidazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateImidazoleRingFormation(BaseScoring):
    """
    Checks for late-stage imidazole ring formation in synthesis routes.
    
    Detects when an imidazole ring (c1ncnc1) is formed via intramolecular cyclization,
    with preference for later stages in the synthesis.
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
            # Later ring formation is better (higher score for higher depth fraction)
            return x * 10  # Scale to 0-10 range
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms an imidazole ring"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Count imidazole rings in products vs reactants
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                              for mol in products if mol is not None)
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                               for mol in reactants if mol is not None)
            
            # Ring formation: more rings in products than reactants
            if self.direction == "formation":
                return product_rings > reactant_rings
            # Ring breaking: fewer rings in products than reactants  
            elif self.direction == "breaking":
                return reactant_rings > product_rings
            
            return False
            
        except Exception:
            return False
