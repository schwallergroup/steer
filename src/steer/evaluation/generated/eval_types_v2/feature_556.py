"""Generated evaluation code for: Chiral auxiliary protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ChiralAuxiliaryProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for the use of chiral auxiliary protecting group strategies.
    Specifically detects the use of tert-butanesulfinyl groups as chiral auxiliaries
    for imine protection/activation.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.chiral = config["parameters"]["chiral"]
        
        # Define SMARTS patterns for detection
        self.tert_butanesulfinyl_pattern = "[S](=O)([C](C)(C)C)[N]"  # tert-butanesulfinyl group
        self.imine_pattern = "[C]=[N]"  # imine functionality
        self.protected_imine_pattern = "[C]=[N][S](=O)[C](C)(C)C"  # protected imine
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not used
        else:
            # Earlier use of protecting group strategy is better (more strategic)
            # Convert depth fraction to 0-10 scale, favoring early use
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves the formation or removal of a chiral auxiliary
        protecting group on an imine.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check for protecting group formation or removal
            return self._detect_protection_strategy(reactants, products) or \
                   self._detect_deprotection_strategy(reactants, products)
                   
        except Exception:
            return False
    
    def _detect_protection_strategy(self, reactants, products) -> bool:
        """
        Detects formation of tert-butanesulfinyl protected imine from free imine.
        """
        # Check if reactants contain free imine and tert-butanesulfinyl reagent
        has_free_imine = any(self._has_free_imine(mol) for mol in reactants)
        has_protecting_reagent = any(self._has_tert_butanesulfinyl_reagent(mol) for mol in reactants)
        
        # Check if products contain protected imine
        has_protected_imine = any(self._has_protected_imine(mol) for mol in products)
        
        return has_free_imine and has_protecting_reagent and has_protected_imine
    
    def _detect_deprotection_strategy(self, reactants, products) -> bool:
        """
        Detects removal of tert-butanesulfinyl protecting group from imine.
        """
        # Check if reactants contain protected imine
        has_protected_imine = any(self._has_protected_imine(mol) for mol in reactants)
        
        # Check if products contain free imine or derived product
        has_free_imine = any(self._has_free_imine(mol) for mol in products)
        has_derived_product = any(self._has_imine_derived_product(mol) for mol in products)
        
        return has_protected_imine and (has_free_imine or has_derived_product)
    
    def _has_free_imine(self, mol) -> bool:
        """Check if molecule contains free imine functionality."""
        if mol is None:
            return False
        imine_pattern = Chem.MolFromSmarts(self.imine_pattern)
        protected_pattern = Chem.MolFromSmarts(self.protected_imine_pattern)
        
        # Has imine but not the protected version
        return mol.HasSubstructMatch(imine_pattern) and not mol.HasSubstructMatch(protected_pattern)
    
    def _has_tert_butanesulfinyl_reagent(self, mol) -> bool:
        """Check if molecule is a tert-butanesulfinyl reagent."""
        if mol is None:
            return False
        pattern = Chem.MolFromSmarts(self.tert_butanesulfinyl_pattern)
        return mol.HasSubstructMatch(pattern)
    
    def _has_protected_imine(self, mol) -> bool:
        """Check if molecule contains tert-butanesulfinyl protected imine."""
        if mol is None:
            return False
        pattern = Chem.MolFromSmarts(self.protected_imine_pattern)
        return mol.HasSubstructMatch(pattern)
    
    def _has_imine_derived_product(self, mol) -> bool:
        """Check if molecule is derived from imine (e.g., reduced to amine)."""
        if mol is None:
            return False
        # Look for primary or secondary amine that could come from imine reduction
        amine_pattern = Chem.MolFromSmarts("[C][NH2,NH1]")
        return mol.HasSubstructMatch(amine_pattern)
