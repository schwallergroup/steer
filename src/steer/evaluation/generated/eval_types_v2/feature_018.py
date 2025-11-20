"""Generated evaluation code for: Benzyl protecting group strategy for phenol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylPhenolProtection(BaseScoring):
    """
    Evaluates benzyl protecting group strategy for phenols.
    Checks if benzyl protection of phenol occurs followed by deprotection via hydrogenation.
    Earlier protection (higher in tree) is scored better.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
        # SMARTS patterns
        self.phenol_pattern = Chem.MolFromSmarts("[OH1][c]")  # Phenol OH
        self.benzyl_ether_pattern = Chem.MolFromSmarts("[CH2][c][O][c]")  # Benzyl ether
        self.benzyl_group_pattern = Chem.MolFromSmarts("[CH2][c]1[cH][cH][cH][cH][cH]1")  # Benzyl group
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better (lower depth fraction)
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                return max(0, 1 - x)  # Convert to 0-1 scale, earlier is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves benzyl protection of phenol"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product has benzyl ether that wasn't in reactants
            product_has_benzyl_ether = product.HasSubstructMatch(self.benzyl_ether_pattern)
            
            if not product_has_benzyl_ether:
                return False
            
            # Check if any reactant has free phenol
            reactant_has_phenol = any(r.HasSubstructMatch(self.phenol_pattern) for r in reactants)
            
            # Check if benzyl group is introduced (present in reactants but not as ether)
            reactant_has_benzyl_reagent = any(
                r.HasSubstructMatch(self.benzyl_group_pattern) and 
                not r.HasSubstructMatch(self.benzyl_ether_pattern) 
                for r in reactants
            )
            
            # This is benzyl protection if:
            # 1. Product has benzyl ether
            # 2. Reactant has free phenol  
            # 3. Benzyl group is introduced as reagent
            return (product_has_benzyl_ether and 
                   reactant_has_phenol and 
                   reactant_has_benzyl_reagent)
                   
        except Exception:
            return False
