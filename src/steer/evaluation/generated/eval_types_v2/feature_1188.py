"""Generated evaluation code for: Benzyl ether protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtectingGroupStrategy(BaseScoring):
    """
    Evaluates whether benzyl ether protecting group strategy is used for alcohols.
    Checks if benzyl ether protection occurs early and deprotection occurs late in the route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fractional")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        self.benzyl_ether_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1")  # Benzyl group
        self.alcohol_pattern = Chem.MolFromSmarts("[OH]")
        self.benzyl_ether_full = Chem.MolFromSmarts("COCc1ccccc1")  # Benzyl ether linkage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        # Reward early protection (higher depth values are better for protection)
        return max(0, min(10, 10 * x))
        
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves benzyl ether protection or deprotection
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        # Parse molecules
        try:
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
        except:
            return False
            
        # Check for protection reaction (alcohol + benzyl halide -> benzyl ether)
        protection_reaction = self._is_protection_reaction(reactants, products)
        
        # Check for deprotection reaction (benzyl ether -> alcohol + benzyl derivative)
        deprotection_reaction = self._is_deprotection_reaction(reactants, products)
        
        return protection_reaction or deprotection_reaction
        
    def _is_protection_reaction(self, reactants, products) -> bool:
        """Check if reaction converts alcohol to benzyl ether"""
        # Look for alcohol in reactants and benzyl ether in products
        has_alcohol_reactant = any(mol.HasSubstructMatch(self.alcohol_pattern) for mol in reactants)
        has_benzyl_reactant = any(mol.HasSubstructMatch(self.benzyl_ether_pattern) for mol in reactants)
        has_benzyl_ether_product = any(mol.HasSubstructMatch(self.benzyl_ether_full) for mol in products)
        
        return has_alcohol_reactant and has_benzyl_reactant and has_benzyl_ether_product
        
    def _is_deprotection_reaction(self, reactants, products) -> bool:
        """Check if reaction converts benzyl ether back to alcohol"""
        # Look for benzyl ether in reactants and alcohol in products
        has_benzyl_ether_reactant = any(mol.HasSubstructMatch(self.benzyl_ether_full) for mol in reactants)
        has_alcohol_product = any(mol.HasSubstructMatch(self.alcohol_pattern) for mol in products)
        has_benzyl_product = any(mol.HasSubstructMatch(self.benzyl_ether_pattern) for mol in products)
        
        return has_benzyl_ether_reactant and has_alcohol_product and has_benzyl_product
