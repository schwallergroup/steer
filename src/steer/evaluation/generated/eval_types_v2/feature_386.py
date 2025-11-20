"""Generated evaluation code for: Trityl protecting group for primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TritylProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for trityl protecting group strategy on primary alcohols.
    Checks if trityl ether protection is used and deprotection occurs in final step.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        # Trityl group pattern - triphenylmethyl
        self.trityl_pattern = Chem.MolFromSmarts("C(c1ccccc1)(c2ccccc2)(c3ccccc3)")
        # Primary alcohol pattern
        self.primary_alcohol_pattern = Chem.MolFromSmarts("[CH2][OH]")
        # Trityl ether pattern (primary alcohol protected with trityl)
        self.trityl_ether_pattern = Chem.MolFromSmarts("[CH2]OC(c1ccccc1)(c2ccccc2)(c3ccccc3)")
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Check if this reaction involves trityl protecting group strategy:
        1. Formation of trityl ether from primary alcohol
        2. Deprotection of trityl ether to primary alcohol
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = Chem.MolFromSmiles(rxn_parts[0])
            reactants_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not products or not reactants:
                return False
            
            # Check for trityl protection: primary alcohol + trityl reagent -> trityl ether
            protection_reaction = self._is_trityl_protection(reactants, products)
            
            # Check for trityl deprotection: trityl ether -> primary alcohol
            deprotection_reaction = self._is_trityl_deprotection(reactants, products)
            
            return protection_reaction or deprotection_reaction
            
        except Exception:
            return False
    
    def _is_trityl_protection(self, reactants, products):
        """Check if reaction is protection of primary alcohol with trityl group"""
        # Look for primary alcohol in reactants and trityl ether in products
        has_primary_alcohol_reactant = any(
            mol.HasSubstructMatch(self.primary_alcohol_pattern) for mol in reactants
        )
        
        has_trityl_ether_product = products.HasSubstructMatch(self.trityl_ether_pattern)
        
        # Check if trityl reagent is present in reactants
        has_trityl_reagent = any(
            mol.HasSubstructMatch(self.trityl_pattern) for mol in reactants
        )
        
        return has_primary_alcohol_reactant and has_trityl_ether_product and has_trityl_reagent
    
    def _is_trityl_deprotection(self, reactants, products):
        """Check if reaction is deprotection of trityl ether to primary alcohol"""
        # Look for trityl ether in reactants and primary alcohol in products
        has_trityl_ether_reactant = any(
            mol.HasSubstructMatch(self.trityl_ether_pattern) for mol in reactants
        )
        
        has_primary_alcohol_product = products.HasSubstructMatch(self.primary_alcohol_pattern)
        
        return has_trityl_ether_reactant and has_primary_alcohol_product
