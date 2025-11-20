"""Generated evaluation code for: Protection deprotection cycle with sulfinyl imine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SulfinylImineProtectionCycle(MultiRxnCondBase):
    """
    Detects protection-deprotection cycles involving tert-butanesulfinyl imine protection.
    Checks if a ketone is converted to N-sulfinyl imine and then deprotected back to imine.
    """
    
    def __init__(self, config):
        self.cycle_present = config.get("cycle_present", True)
        self.protection_pattern = "[S](=O)(C(C)(C)C)[N]=[C]"  # tert-butanesulfinyl imine
        self.ketone_pattern = "[C]=O"  # ketone
        self.imine_pattern = "[C]=[N]"  # imine without sulfinyl protection
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Look for protection-deprotection cycle
        has_protection = False
        has_deprotection = False
        
        for rxn in reactions:
            if self.detect_sulfinyl_protection(rxn):
                has_protection = True
            if self.detect_sulfinyl_deprotection(rxn):
                has_deprotection = True
        
        # Check if cycle is present as expected
        cycle_detected = has_protection and has_deprotection
        condition = cycle_detected == self.cycle_present
        
        return condition, len(reactions)
    
    def detect_sulfinyl_protection(self, rxn):
        """Detect formation of tert-butanesulfinyl imine from ketone"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if ketone in reactants and sulfinyl imine in products
        has_ketone_reactant = any(self.has_substructure(r, self.ketone_pattern) for r in reactants)
        has_sulfinyl_product = any(self.has_substructure(p, self.protection_pattern) for p in products)
        
        return has_ketone_reactant and has_sulfinyl_product
    
    def detect_sulfinyl_deprotection(self, rxn):
        """Detect removal of tert-butanesulfinyl group to form free imine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if sulfinyl imine in reactants and free imine in products
        has_sulfinyl_reactant = any(self.has_substructure(r, self.protection_pattern) for r in reactants)
        has_imine_product = any(self.has_substructure(p, self.imine_pattern) for p in products)
        
        # Make sure the imine product doesn't have sulfinyl protection
        has_protected_product = any(self.has_substructure(p, self.protection_pattern) for p in products)
        
        return has_sulfinyl_reactant and has_imine_product and not has_protected_product
    
    def has_substructure(self, smiles, pattern):
        """Check if molecule contains the given substructure pattern"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
            return mol.HasSubstructMatch(pattern_mol)
        except:
            return False
