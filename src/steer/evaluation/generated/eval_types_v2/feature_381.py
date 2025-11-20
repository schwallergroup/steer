"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring formation occurs at a late stage in the synthesis.
    Detects ring formation by checking if the ring pattern appears in products but not reactants.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early" or "late"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        else:  # early
            return x  # Earlier formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction step involves the target ring formation/break"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Filter out None molecules
            products = [p for p in products if p is not None]
            reactants = [r for r in reactants if r is not None]
            
            # Create pattern matcher
            pattern = Chem.MolFromSmarts(self.ring_smarts)
            if pattern is None:
                return False
            
            # Check for ring pattern in products and reactants
            ring_in_products = any(mol.HasSubstructMatch(pattern) for mol in products)
            ring_in_reactants = any(mol.HasSubstructMatch(pattern) for mol in reactants)
            
            if self.direction == "formation":
                # Ring formation: pattern appears in products but not in reactants
                return ring_in_products and not ring_in_reactants
            else:  # break
                # Ring break: pattern appears in reactants but not in products
                return ring_in_reactants and not ring_in_products
                
        except Exception:
            return False
