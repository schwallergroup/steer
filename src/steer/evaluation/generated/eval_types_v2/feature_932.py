"""Generated evaluation code for: Late pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring structure is formed late in the synthesis route.
    Returns higher scores for ring formations that occur closer to the end of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        elif self.timing == "early":
            return x  # Earlier formation gets higher score
        else:
            return 1 if x >= 0 else 0  # Just check if it happens
    
    def hit_condition(self, d):
        """Check if the specified ring is formed in this reaction step"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            # Filter out None molecules
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            # Create SMARTS pattern for ring detection
            ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if ring_pattern is None:
                return False
            
            # Check if ring is present in products but not in reactants (formation)
            if self.direction == "formation":
                ring_in_products = any(mol.HasSubstructMatch(ring_pattern) for mol in products)
                ring_in_reactants = any(mol.HasSubstructMatch(ring_pattern) for mol in reactants)
                
                return ring_in_products and not ring_in_reactants
            
            # Check if ring is present in reactants but not in products (breaking)
            elif self.direction == "breaking":
                ring_in_products = any(mol.HasSubstructMatch(ring_pattern) for mol in products)
                ring_in_reactants = any(mol.HasSubstructMatch(ring_pattern) for mol in reactants)
                
                return ring_in_reactants and not ring_in_products
            
            return False
            
        except Exception:
            return False
