"""Generated evaluation code for: Late stage piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring structure is formed late in the synthesis route.
    Higher scores are given when the ring formation occurs closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Late-stage formation is better (lower depth fraction)
        else:
            return x  # Early-stage formation is better (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction involves formation of the target ring structure.
        For formation: ring absent in reactants but present in products.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = [Chem.MolFromSmiles(p) for p in rxn[0].split(".")]
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check if ring is present in products
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products if mol is not None)
            
            # Check if ring is present in reactants
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants if mol is not None)
            
            if self.direction == "formation":
                # Ring formation: ring in products but not in reactants
                return ring_in_products and not ring_in_reactants
            else:
                # Ring breaking: ring in reactants but not in products
                return ring_in_reactants and not ring_in_products
                
        except Exception:
            return False
