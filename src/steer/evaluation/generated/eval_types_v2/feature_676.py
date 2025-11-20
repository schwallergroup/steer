"""Generated evaluation code for: Ellman auxiliary protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EllmanAuxiliaryStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of Ellman auxiliary protecting group strategy.
    Checks if tert-butanesulfinamide is used to protect imine intermediates.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        # Ellman auxiliary (tert-butanesulfinamide) pattern
        self.sulfinyl_pattern = Chem.MolFromSmarts("CC(C)(C)S(=O)N")
        # Imine pattern to check for protection
        self.imine_pattern = Chem.MolFromSmarts("C=N")
        # Protected imine pattern (sulfinyl imine)
        self.protected_imine_pattern = Chem.MolFromSmarts("CC(C)(C)S(=O)N=C")
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Strategy not used
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Check if reaction involves Ellman auxiliary protection/deprotection.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[0].split(".") if p]
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".") if r]
            
            if not products or not reactants:
                return False
            
            # Check for protection reaction: imine + sulfinyl reagent -> protected imine
            protection_reaction = self._is_protection_reaction(reactants, products)
            
            # Check for deprotection reaction: protected imine -> imine + sulfinyl
            deprotection_reaction = self._is_deprotection_reaction(reactants, products)
            
            # Check for reactions using protected intermediate
            using_protected_intermediate = self._uses_protected_intermediate(reactants, products)
            
            return protection_reaction or deprotection_reaction or using_protected_intermediate
            
        except Exception:
            return False
    
    def _is_protection_reaction(self, reactants, products):
        """Check if this is an imine protection reaction using Ellman auxiliary."""
        # Look for sulfinyl reagent in reactants and imine in reactants
        has_sulfinyl_reagent = any(mol.HasSubstructMatch(self.sulfinyl_pattern) for mol in reactants if mol)
        has_imine_reactant = any(mol.HasSubstructMatch(self.imine_pattern) for mol in reactants if mol)
        
        # Look for protected imine in products
        has_protected_product = any(mol.HasSubstructMatch(self.protected_imine_pattern) for mol in products if mol)
        
        return has_sulfinyl_reagent and has_imine_reactant and has_protected_product
    
    def _is_deprotection_reaction(self, reactants, products):
        """Check if this is a deprotection reaction removing Ellman auxiliary."""
        # Look for protected imine in reactants
        has_protected_reactant = any(mol.HasSubstructMatch(self.protected_imine_pattern) for mol in reactants if mol)
        
        # Look for free imine or amine in products (after reduction/hydrolysis)
        has_deprotected_product = any(mol.HasSubstructMatch(self.imine_pattern) or 
                                    mol.HasSubstructMatch(Chem.MolFromSmarts("CN")) for mol in products if mol)
        
        # Look for sulfinyl byproduct
        has_sulfinyl_byproduct = any(mol.HasSubstructMatch(self.sulfinyl_pattern) for mol in products if mol)
        
        return has_protected_reactant and (has_deprotected_product or has_sulfinyl_byproduct)
    
    def _uses_protected_intermediate(self, reactants, products):
        """Check if reaction uses a protected imine intermediate."""
        # Check if protected imine is present as reactant or product
        has_protected_reactant = any(mol.HasSubstructMatch(self.protected_imine_pattern) for mol in reactants if mol)
        has_protected_product = any(mol.HasSubstructMatch(self.protected_imine_pattern) for mol in products if mol)
        
        return has_protected_reactant or has_protected_product
