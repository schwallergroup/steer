"""Generated evaluation code for: Ellman sulfinamide protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EllmanSulfinamideStrategy(BaseScoring):
    """
    Evaluates the use of Ellman sulfinamide protecting group strategy.
    Checks for the presence of tert-butanesulfinyl protection on imine substrates
    for stabilization purposes.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not used
        else:
            # Earlier use of protecting group strategy is better
            return 1 - x
    
    def hit_condition(self, d):
        """
        Check if a reaction involves Ellman sulfinamide protecting group strategy
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for protection: formation of tert-butanesulfinyl imine
            if self._is_protection_reaction(reactants, products):
                return True
                
            # Check for deprotection: removal of tert-butanesulfinyl group
            if self._is_deprotection_reaction(reactants, products):
                return True
                
            return False
            
        except Exception:
            return False
    
    def _is_protection_reaction(self, reactants, products):
        """Check if reaction involves protection with tert-butanesulfinyl group"""
        # Pattern for tert-butanesulfinamide reagent: (CH3)3C-S(=O)-NH2
        sulfinamide_pattern = "[CH3][C]([CH3])([CH3])[S](=O)[NH2]"
        
        # Pattern for imine substrate: C=N
        imine_pattern = "[CH]=[NH]"
        
        # Pattern for protected product: C=N-S(=O)-C(CH3)3
        protected_imine_pattern = "[CH]=[N][S](=O)[C]([CH3])([CH3])[CH3]"
        
        try:
            # Check reactants for sulfinamide and imine
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            has_sulfinamide = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(sulfinamide_pattern))
                for mol in reactant_mols
            )
            
            has_imine_reactant = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(imine_pattern))
                for mol in reactant_mols
            )
            
            has_protected_product = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(protected_imine_pattern))
                for mol in product_mols
            )
            
            return has_sulfinamide and has_imine_reactant and has_protected_product
            
        except Exception:
            return False
    
    def _is_deprotection_reaction(self, reactants, products):
        """Check if reaction involves deprotection of tert-butanesulfinyl group"""
        # Pattern for protected imine: C=N-S(=O)-C(CH3)3
        protected_imine_pattern = "[CH]=[N][S](=O)[C]([CH3])([CH3])[CH3]"
        
        # Pattern for deprotected product: C=N or C-N
        deprotected_pattern = "[CH]=[NH],[CH][NH2]"
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            has_protected_reactant = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(protected_imine_pattern))
                for mol in reactant_mols
            )
            
            has_deprotected_product = any(
                mol and any(
                    mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                    for pattern in deprotected_pattern.split(",")
                )
                for mol in product_mols
            )
            
            return has_protected_reactant and has_deprotected_product
            
        except Exception:
            return False
