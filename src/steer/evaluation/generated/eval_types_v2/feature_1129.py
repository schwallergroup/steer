"""Generated evaluation code for: Early stage pyrimidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrimidineFormation(BaseScoring):
    """
    Evaluates whether pyrimidine ring formation occurs early in the synthesis route.
    Detects cyclocondensation reactions that form pyrimidine rings and scores based on timing.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "n1cnccc1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Earlier formation gets higher score
            else:  # late
                return x  # Later formation gets higher score
                
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a pyrimidine ring"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Count pyrimidine rings in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                               for mol in reactants)
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                              for mol in products)
            
            # Check for ring formation (more rings in products than reactants)
            if self.direction == "formation":
                return product_rings > reactant_rings
            else:  # breaking
                return reactant_rings > product_rings
                
        except (KeyError, AttributeError, ValueError):
            return False
