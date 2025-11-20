"""Generated evaluation code for: Late thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation.
    Checks if thiazole rings are formed in the last few steps rather than
    using pre-formed thiazole starting materials.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.step_threshold = config["parameters"]["step_threshold"]  # 3
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole ring formation not found
        
        if self.timing == "late":
            # Reward formation in last few steps (lower depth fraction is better)
            if x <= (self.step_threshold / 10.0):  # Convert step threshold to fraction
                return 10  # Perfect score for very late formation
            else:
                return max(0, 10 - (x * 10))  # Linear decrease as formation gets earlier
        else:
            return 10 - abs(x * 10 - self.step_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a thiazole ring by comparing
        reactants and products for thiazole substructure presence.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse products
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            products = [p for p in products if p is not None]
            
            # Parse reactants
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            # Check if thiazole is present in products
            thiazole_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            # Check if thiazole is present in reactants
            thiazole_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            
            # Ring formation occurs if thiazole is in products but not in reactants
            return thiazole_in_products and not thiazole_in_reactants
            
        except Exception:
            return False
