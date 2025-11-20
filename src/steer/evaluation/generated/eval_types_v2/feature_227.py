"""Generated evaluation code for: THP protecting group strategy for secondary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class THPProtectingGroupStrategy(BaseScoring):
    """
    Evaluates the use of THP (tetrahydropyranyl) protecting group strategy 
    for secondary alcohols during synthesis routes.
    """
    
    def __init__(self, config: Dict):
        self.usage = config["parameters"].get("usage", "minimal")
        
        # SMARTS patterns for THP group and secondary alcohol
        self.thp_pattern = Chem.MolFromSmarts("[CH]1OCCCC1")  # THP ring
        self.secondary_alcohol_pattern = Chem.MolFromSmarts("[CH]([OH])")  # Secondary alcohol
        self.thp_ether_pattern = Chem.MolFromSmarts("[CH]1OCCCC1O[CH]")  # THP ether linkage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # THP protection not used
        
        if self.usage == "minimal":
            # Reward early protection (lower depth values are better)
            return max(0, 10 - (x * 10))
        else:
            # Standard depth-based scoring
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves THP protection of a secondary alcohol
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".") if smi.strip()]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".") if smi.strip()]
            
            if not reactants or not products:
                return False
            
            # Check if reaction involves THP protection
            return self._is_thp_protection_reaction(reactants, products)
            
        except Exception:
            return False
    
    def _is_thp_protection_reaction(self, reactants, products) -> bool:
        """
        Determine if this is a THP protection reaction by checking:
        1. Reactants contain secondary alcohol and THP precursor
        2. Products contain THP ether
        """
        # Check reactants for secondary alcohol
        has_sec_alcohol_reactant = any(
            mol and mol.HasSubstructMatch(self.secondary_alcohol_pattern) 
            for mol in reactants if mol is not None
        )
        
        # Check reactants for THP or THP precursor (3,4-dihydro-2H-pyran)
        dhp_pattern = Chem.MolFromSmarts("C1=CCCCO1")  # 3,4-dihydro-2H-pyran (THP precursor)
        has_thp_precursor = any(
            mol and (mol.HasSubstructMatch(self.thp_pattern) or mol.HasSubstructMatch(dhp_pattern))
            for mol in reactants if mol is not None
        )
        
        # Check products for THP ether formation
        has_thp_ether_product = any(
            mol and mol.HasSubstructMatch(self.thp_ether_pattern)
            for mol in products if mol is not None
        )
        
        # Also check for reduced secondary alcohol count in products
        sec_alcohol_count_reactants = sum(
            len(mol.GetSubstructMatches(self.secondary_alcohol_pattern)) if mol else 0
            for mol in reactants
        )
        
        sec_alcohol_count_products = sum(
            len(mol.GetSubstructMatches(self.secondary_alcohol_pattern)) if mol else 0
            for mol in products
        )
        
        alcohol_protected = sec_alcohol_count_products < sec_alcohol_count_reactants
        
        # Return True if this looks like THP protection of secondary alcohol
        return (has_sec_alcohol_reactant and 
                has_thp_precursor and 
                has_thp_ether_product and 
                alcohol_protected)
