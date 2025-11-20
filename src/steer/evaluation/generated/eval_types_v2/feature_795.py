"""Generated evaluation code for: Late imidazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateImidazoleRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when imidazole ring formation occurs.
    Rewards routes where imidazole rings are formed late in the synthesis.
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
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score
            elif self.timing == "early":
                return x  # Earlier formation gets higher score
            else:
                return 0.5  # Default scoring
    
    def hit_condition(self, d):
        """Check if imidazole ring formation occurs in this reaction step"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if self.direction == "formation":
                # Check if ring is formed: absent in reactants but present in products
                reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return not reactant_has_ring and product_has_ring
            
            elif self.direction == "breaking":
                # Check if ring is broken: present in reactants but absent in products
                reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                
                return reactant_has_ring and not product_has_ring
            
            return False
            
        except (KeyError, ValueError, AttributeError):
            return False
