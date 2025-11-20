"""Generated evaluation code for: Benzyl ether protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for the use of benzyl ether protecting group strategy.
    Checks if a phenol is protected with a benzyl group and later deprotected.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # SMARTS patterns for benzyl ether protection and deprotection
        self.benzyl_ether_pattern = "[OH1]-[CH2]-c1ccccc1"  # Benzyl ether
        self.phenol_pattern = "[OH1]-c1ccccc1"  # Phenol
        self.benzyl_group_pattern = "[CH2]-c1ccccc1"  # Benzyl group
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)
    
    def hit_condition(self, d):
        """
        Check if this reaction involves benzyl ether protecting group strategy:
        1. Protection: phenol + benzyl reagent -> benzyl ether
        2. Deprotection: benzyl ether -> phenol (typically via hydrogenation)
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check for protection reaction: phenol -> benzyl ether
            if self._is_protection_reaction(reactants, products):
                return True
                
            # Check for deprotection reaction: benzyl ether -> phenol
            if self._is_deprotection_reaction(reactants, products):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_protection_reaction(self, reactants, products):
        """Check if reaction is phenol protection with benzyl group"""
        # Look for phenol in reactants and benzyl ether in products
        has_phenol_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern))
            for mol in reactants if mol
        )
        
        has_benzyl_reagent = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_group_pattern))
            for mol in reactants if mol
        )
        
        has_benzyl_ether_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern))
            for mol in products if mol
        )
        
        return has_phenol_reactant and has_benzyl_reagent and has_benzyl_ether_product
    
    def _is_deprotection_reaction(self, reactants, products):
        """Check if reaction is benzyl ether deprotection to phenol"""
        # Look for benzyl ether in reactants and phenol in products
        has_benzyl_ether_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern))
            for mol in reactants if mol
        )
        
        has_phenol_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern))
            for mol in products if mol
        )
        
        # Check for typical deprotection conditions (H2, Pd/C, etc.)
        # This can be inferred from the presence of small molecules or reaction metadata
        has_hydrogenation_conditions = any(
            Chem.MolToSmiles(mol) in ["[H][H]", "O"] for mol in products if mol
        )
        
        return has_benzyl_ether_reactant and has_phenol_product
