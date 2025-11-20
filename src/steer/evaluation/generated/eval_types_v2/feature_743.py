"""Generated evaluation code for: Protection-deprotection cycle on sugar tertiary hydroxyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectionDeprotectionCycle(MultiRxnCondBase):
    """
    Evaluates protection-deprotection cycles on sugar tertiary hydroxyl groups.
    Checks for TBDMS protection followed by deprotection in the same synthetic sequence.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "TBDMS")
        self.position = config.get("position", "tertiary_hydroxyl")
        self.strategy = config.get("strategy", "cyclic")
        
        # TBDMS patterns for protection and deprotection
        self.tbdms_protected = "[OH1]Si(C(C)(C)C)(C)(C)"  # TBDMS-protected hydroxyl
        self.tertiary_oh = "[CH0]([OH1])"  # Tertiary carbon with OH
        self.sugar_pattern = "[CH1]1[OH0,OH1][CH1]([OH0,OH1])[CH1]([OH0,OH1])[CH1]([OH0,OH1])[OH0]1"  # Sugar ring
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = False
        deprotection_found = False
        protection_depth = -1
        deprotection_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_tbdms_protection(rxn):
                protection_found = True
                protection_depth = i
            elif self.detect_tbdms_deprotection(rxn):
                deprotection_found = True
                deprotection_depth = i
        
        # Check if we have a complete cycle (protection followed by deprotection)
        cycle_complete = (protection_found and deprotection_found and 
                         protection_depth < deprotection_depth)
        
        if cycle_complete:
            # Return depth as the deprotection step (completion of cycle)
            return True, deprotection_depth
        
        return False, len(reactions)
    
    def detect_tbdms_protection(self, rxn):
        """Detect TBDMS protection of tertiary hydroxyl on sugar"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check for tertiary OH in sugar substrate in reactants
        sugar_reactant = None
        for mol in reactants:
            if self.has_sugar_with_tertiary_oh(mol):
                sugar_reactant = mol
                break
        
        if not sugar_reactant:
            return False
        
        # Check for TBDMS-protected sugar in products
        for mol in products:
            if self.has_tbdms_protected_sugar(mol):
                return True
        
        return False
    
    def detect_tbdms_deprotection(self, rxn):
        """Detect TBDMS deprotection revealing tertiary hydroxyl on sugar"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check for TBDMS-protected sugar in reactants
        tbdms_reactant = None
        for mol in reactants:
            if self.has_tbdms_protected_sugar(mol):
                tbdms_reactant = mol
                break
        
        if not tbdms_reactant:
            return False
        
        # Check for deprotected sugar with tertiary OH in products
        for mol in products:
            if self.has_sugar_with_tertiary_oh(mol):
                return True
        
        return False
    
    def has_sugar_with_tertiary_oh(self, mol):
        """Check if molecule contains sugar ring with tertiary hydroxyl"""
        if not mol:
            return False
        
        sugar_pattern_mol = Chem.MolFromSmarts(self.sugar_pattern)
        tertiary_oh_mol = Chem.MolFromSmarts(self.tertiary_oh)
        
        if not sugar_pattern_mol or not tertiary_oh_mol:
            return False
        
        return (mol.HasSubstructMatch(sugar_pattern_mol) and 
                mol.HasSubstructMatch(tertiary_oh_mol))
    
    def has_tbdms_protected_sugar(self, mol):
        """Check if molecule contains sugar ring with TBDMS-protected hydroxyl"""
        if not mol:
            return False
        
        sugar_pattern_mol = Chem.MolFromSmarts(self.sugar_pattern)
        tbdms_pattern_mol = Chem.MolFromSmarts(self.tbdms_protected)
        
        if not sugar_pattern_mol or not tbdms_pattern_mol:
            return False
        
        return (mol.HasSubstructMatch(sugar_pattern_mol) and 
                mol.HasSubstructMatch(tbdms_pattern_mol))
