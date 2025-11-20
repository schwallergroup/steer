"""Generated evaluation code for: Early stage indole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyStageIndoleFormation(BaseScoring):
    """
    Evaluates whether indole ring formation occurs in the early stages of synthesis.
    Uses SMARTS pattern matching to detect indole formation reactions and checks
    if they occur before the specified stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.stage_threshold = config["parameters"]["stage_threshold"]
        self.indole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Indole formation doesn't happen
        
        if self.timing == "early":
            if x <= self.stage_threshold:
                return 1.0  # Perfect score for early formation
            else:
                # Penalize late formation, score decreases linearly
                return max(0, 1.0 - (x - self.stage_threshold) / (1.0 - self.stage_threshold))
        else:
            # For other timing preferences, adjust scoring accordingly
            return 1.0 - abs(x - self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if the reaction forms an indole ring by detecting indole presence
        in products but absence in at least one reactant.
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
            
            # Check if any product contains indole
            product_has_indole = any(mol.HasSubstructMatch(self.indole_pattern) for mol in products)
            
            if not product_has_indole:
                return False
            
            # Check if indole is newly formed (not present in all reactants)
            reactant_indole_count = sum(1 for mol in reactants if mol.HasSubstructMatch(self.indole_pattern))
            product_indole_count = sum(len(mol.GetSubstructMatches(self.indole_pattern)) for mol in products)
            
            # Indole formation occurs if product has more indole rings than reactants
            return product_indole_count > reactant_indole_count
            
        except Exception:
            return False
