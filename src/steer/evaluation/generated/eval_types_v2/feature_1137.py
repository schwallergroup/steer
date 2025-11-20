"""Generated evaluation code for: Late thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleRingFormation(BaseScoring):
    """
    Evaluates routes for late-stage thiazole ring formation.
    Checks if thiazole rings (c1scnc1) are formed in the later stages of the synthesis route.
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
            # Late-stage formation is better, so higher depth gets higher score
            if self.timing == "late":
                return x  # Later formation gets higher score
            else:  # early timing
                return 1 - x  # Earlier formation gets higher score
    
    def hit_condition(self, d):
        """Check if thiazole ring formation occurs in this reaction step"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            product_mol = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Count thiazole rings in products
            product_thiazole_count = len(product_mol.GetSubstructMatches(self.ring_pattern))
            
            # Count thiazole rings in all reactants
            reactant_thiazole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactant_mols
            )
            
            # Check if thiazole ring formation occurred
            if self.direction == "formation":
                return product_thiazole_count > reactant_thiazole_count
            elif self.direction == "breaking":
                return reactant_thiazole_count > product_thiazole_count
            else:
                return False
                
        except Exception:
            return False
