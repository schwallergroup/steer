"""Generated evaluation code for: Cbz protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for proper Cbz protecting group strategy.
    Checks for presence of Cbz protection on secondary amines and appropriate
    deprotection via hydrogenolysis before final functionalization.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Cbz")
        self.functional_group = config.get("functional_group", "secondary_amine")
        self.deprotection_method = config.get("deprotection_method", "hydrogenolysis")
        self.require_protection = config.get("require_protection", True)
        self.require_deprotection = config.get("require_deprotection", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_cbz_protection = any(self.detect_cbz_protection(r) for r in reactions)
        has_cbz_deprotection = any(self.detect_cbz_deprotection(r) for r in reactions)
        
        # Check if protection occurs before deprotection in the route
        protection_depth = -1
        deprotection_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_cbz_protection(rxn) and protection_depth == -1:
                protection_depth = i
            if self.detect_cbz_deprotection(rxn) and deprotection_depth == -1:
                deprotection_depth = i
        
        # Strategy is correct if:
        # 1. Protection is used when required
        # 2. Deprotection is used when required  
        # 3. Protection occurs before deprotection (if both present)
        condition_met = True
        
        if self.require_protection and not has_cbz_protection:
            condition_met = False
            
        if self.require_deprotection and not has_cbz_deprotection:
            condition_met = False
            
        if (protection_depth >= 0 and deprotection_depth >= 0 and 
            protection_depth >= deprotection_depth):
            condition_met = False  # Protection should come before deprotection
        
        return condition_met, len(reactions)
    
    def detect_cbz_protection(self, rxn):
        """Detect Cbz protection of secondary amine"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        # Cbz protecting group pattern (benzyloxycarbonyl)
        cbz_pattern = Chem.MolFromSmarts("NC(=O)OCc1ccccc1")
        protected_amine_pattern = Chem.MolFromSmarts("[NH1]C(=O)OCc1ccccc1")
        
        # Check if product has Cbz-protected amine that wasn't in reactants
        if prod_mol and prod_mol.HasSubstructMatch(protected_amine_pattern):
            # Check that reactants had free secondary amine
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            for react_mol in react_mols:
                if (react_mol and react_mol.HasSubstructMatch(free_amine_pattern) and 
                    not react_mol.HasSubstructMatch(protected_amine_pattern)):
                    return True
        
        return False
    
    def detect_cbz_deprotection(self, rxn):
        """Detect Cbz deprotection via hydrogenolysis"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        protected_amine_pattern = Chem.MolFromSmarts("[NH1]C(=O)OCc1ccccc1")
        free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
        # Check if reactant has Cbz-protected amine and product has free amine
        has_protected_reactant = any(mol and mol.HasSubstructMatch(protected_amine_pattern) 
                                   for mol in react_mols if mol)
        has_free_product = prod_mol and prod_mol.HasSubstructMatch(free_amine_pattern)
        
        # Also check for presence of hydrogen or hydrogenation conditions
        has_hydrogen = any("H" in rxn[1] for _ in [1])  # H2 or [H]
        
        return has_protected_reactant and has_free_product and has_hydrogen
