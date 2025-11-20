"""Generated evaluation code for: Methoxy protecting group for phenol functionality"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MethoxyPhenolProtection(BaseScoring):
    """
    Evaluates methoxy protecting group strategy for phenol functionality.
    Checks if methoxy-protected phenol is deprotected at the specified timing in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.deprotection_timing = config["parameters"]["deprotection_timing"]  # "early", "mid", "late"
        self.timing_thresholds = {
            "early": 0.8,   # Within first 20% of route
            "mid": 0.5,     # Around middle of route  
            "late": 0.2     # Within last 20% of route
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Deprotection doesn't happen
        
        target_threshold = self.timing_thresholds[self.deprotection_timing]
        
        if self.deprotection_timing == "late":
            # For late deprotection, reward lower depth fractions (closer to end)
            if x <= target_threshold:
                return 10
            else:
                return max(0, 10 - 50 * (x - target_threshold))
        elif self.deprotection_timing == "early":
            # For early deprotection, reward higher depth fractions (closer to start)
            if x >= target_threshold:
                return 10
            else:
                return max(0, 10 - 50 * (target_threshold - x))
        else:  # mid
            # For mid deprotection, reward values around 0.5
            distance_from_mid = abs(x - 0.5)
            if distance_from_mid <= 0.1:
                return 10
            else:
                return max(0, 10 - 50 * distance_from_mid)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves methoxy deprotection of a phenol."""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactant = Chem.MolFromSmiles(rxn_parts[0])
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            if not reactant or not all(products):
                return False
            
            # Check for methoxy-phenol in reactant
            methoxy_phenol_pattern = Chem.MolFromSmarts("[OH1]c1ccc(OC)cc1")  # para-methoxyphenol
            methoxy_phenol_pattern2 = Chem.MolFromSmarts("[OH1]c1cc(OC)ccc1")  # meta-methoxyphenol
            methoxy_phenol_pattern3 = Chem.MolFromSmarts("[OH1]c1ccccc1OC")    # ortho-methoxyphenol
            
            has_methoxy_phenol_reactant = (reactant.HasSubstructMatch(methoxy_phenol_pattern) or
                                         reactant.HasSubstructMatch(methoxy_phenol_pattern2) or
                                         reactant.HasSubstructMatch(methoxy_phenol_pattern3))
            
            if not has_methoxy_phenol_reactant:
                return False
            
            # Check for free phenol in products (deprotected)
            phenol_pattern = Chem.MolFromSmarts("[OH1]c1ccccc1")
            has_free_phenol_product = any(p.HasSubstructMatch(phenol_pattern) for p in products)
            
            # Check for methanol or methyl-containing byproduct (indicating demethylation)
            methanol_pattern = Chem.MolFromSmarts("CO")
            has_methanol_product = any(p.HasSubstructMatch(methanol_pattern) for p in products)
            
            return has_free_phenol_product and has_methanol_product
            
        except Exception:
            return False
