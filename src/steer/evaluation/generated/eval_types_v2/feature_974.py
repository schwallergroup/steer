"""Generated evaluation code for: Temporary sulfinamide protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SulfinamideProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for the use of temporary sulfinamide protecting groups
    on imine functionality. Detects N-tert-butanesulfinamide protection/deprotection
    strategies in synthetic routes.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # Sulfinamide protecting group patterns
        self.sulfinamide_pattern = Chem.MolFromSmarts("[NH1][S](=O)[C]")  # N-sulfinamide
        self.tert_butyl_sulfinamide_pattern = Chem.MolFromSmarts("[NH1][S](=O)C(C)(C)C")  # N-tert-butylsulfinamide
        self.imine_pattern = Chem.MolFromSmarts("[#6]=[NH1]")  # Imine C=N
        self.protected_imine_pattern = Chem.MolFromSmarts("[#6][NH1][S](=O)")  # Protected imine
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth) * 0.2)
    
    def hit_condition(self, d):
        """
        Checks if a reaction involves sulfinamide protection/deprotection of an imine
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for protection: imine + sulfinamide -> protected imine
            protection_reaction = self._is_protection_reaction(reactants, products)
            
            # Check for deprotection: protected imine -> imine + sulfinamide
            deprotection_reaction = self._is_deprotection_reaction(reactants, products)
            
            return protection_reaction or deprotection_reaction
            
        except Exception:
            return False
    
    def _is_protection_reaction(self, reactants, products):
        """Check if reaction is imine protection with sulfinamide"""
        # Look for imine in reactants and sulfinamide reagent
        has_imine_reactant = any(mol.HasSubstructMatch(self.imine_pattern) for mol in reactants)
        has_sulfinamide_reagent = any(mol.HasSubstructMatch(self.sulfinamide_pattern) for mol in reactants)
        
        # Look for protected imine in products
        has_protected_product = any(mol.HasSubstructMatch(self.protected_imine_pattern) for mol in products)
        
        return has_imine_reactant and has_sulfinamide_reagent and has_protected_product
    
    def _is_deprotection_reaction(self, reactants, products):
        """Check if reaction is sulfinamide deprotection to regenerate imine"""
        # Look for protected imine in reactants
        has_protected_reactant = any(mol.HasSubstructMatch(self.protected_imine_pattern) for mol in reactants)
        
        # Look for imine in products and sulfinamide byproduct
        has_imine_product = any(mol.HasSubstructMatch(self.imine_pattern) for mol in products)
        has_sulfinamide_byproduct = any(mol.HasSubstructMatch(self.sulfinamide_pattern) for mol in products)
        
        return has_protected_reactant and (has_imine_product or has_sulfinamide_byproduct)
