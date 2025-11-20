"""Generated evaluation code for: Early Fischer indole core assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FischerIndoleCoreAssembly(BaseScoring):
    """
    Evaluates whether Fischer indole core assembly occurs early in the synthesis route.
    Detects formation of indole-like tricyclic structures and rewards early-stage assembly.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Core assembly doesn't happen
        
        if self.timing == "early":
            return 1 - x  # Earlier is better, score approaches 1 for x near 0
        else:
            return x  # Later is better
    
    def hit_condition(self, d) -> bool:
        """
        Detects Fischer indole core formation by checking if the indole pattern
        is formed (present in products but not in all reactants).
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse products and reactants
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            products = [p for p in products if p is not None]
            reactants = [r for r in reactants if r is not None]
            
            if not products or not reactants:
                return False
            
            # Check if indole core is present in products
            indole_in_products = any(p.HasSubstructMatch(self.ring_pattern) for p in products)
            
            if not indole_in_products:
                return False
            
            # For formation: indole core should be absent in all reactants but present in products
            if self.direction == "formation":
                indole_in_reactants = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
                return not indole_in_reactants
            
            # For breaking: indole core should be present in reactants but absent in products
            elif self.direction == "breaking":
                indole_in_reactants = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
                indole_in_products = any(p.HasSubstructMatch(self.ring_pattern) for p in products)
                return indole_in_reactants and not indole_in_products
            
            return False
            
        except (KeyError, ValueError, AttributeError):
            return False
