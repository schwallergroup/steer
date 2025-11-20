"""Generated evaluation code for: Protecting group cycling secondary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes that use protecting group cycling for secondary alcohols.
    Checks if a secondary alcohol is protected (e.g., with acetate) and then deprotected.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "secondary_alcohol")
        self.protecting_groups = config.get("protecting_groups", ["acetate"])
        self.require_cycles = config.get("protection_deprotection_cycles", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection and deprotection events
        protection_found = False
        deprotection_found = False
        
        for rxn in reactions:
            if self.detect_protection(rxn):
                protection_found = True
            if self.detect_deprotection(rxn):
                deprotection_found = True
        
        # Condition is met if we find both protection and deprotection
        condition = protection_found and deprotection_found if self.require_cycles else (protection_found or deprotection_found)
        
        return condition, len(reactions)
    
    def detect_protection(self, rxn):
        """Detect protection of secondary alcohol with specified protecting groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Secondary alcohol pattern: C[CH]([OH])C
        sec_alcohol_pattern = Chem.MolFromSmarts("[C][CH]([OH])[C]")
        
        # Check if reactants contain secondary alcohol
        has_sec_alcohol_reactant = any(mol.HasSubstructMatch(sec_alcohol_pattern) for mol in reactants)
        
        if not has_sec_alcohol_reactant:
            return False
        
        # Check for protecting group formation
        for pg in self.protecting_groups:
            if pg == "acetate":
                # Acetate protection: C[CH](OC(=O)C)C
                protected_pattern = Chem.MolFromSmarts("[C][CH](OC(=O)[C])[C]")
                has_protected_product = any(mol.HasSubstructMatch(protected_pattern) for mol in products)
                if has_protected_product:
                    return True
        
        return False
    
    def detect_deprotection(self, rxn):
        """Detect deprotection of secondary alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Secondary alcohol pattern: C[CH]([OH])C
        sec_alcohol_pattern = Chem.MolFromSmarts("[C][CH]([OH])[C]")
        
        # Check if products contain free secondary alcohol
        has_sec_alcohol_product = any(mol.HasSubstructMatch(sec_alcohol_pattern) for mol in products)
        
        if not has_sec_alcohol_product:
            return False
        
        # Check for protecting group removal
        for pg in self.protecting_groups:
            if pg == "acetate":
                # Acetate protected: C[CH](OC(=O)C)C
                protected_pattern = Chem.MolFromSmarts("[C][CH](OC(=O)[C])[C]")
                has_protected_reactant = any(mol.HasSubstructMatch(protected_pattern) for mol in reactants)
                if has_protected_reactant:
                    return True
        
        return False
