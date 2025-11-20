"""Generated evaluation code for: Deprotection protection cycle on phenol group"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PhenolProtectionCycle(MultiRxnCondBase):
    """
    Detects deprotection-protection cycles on phenol groups, specifically 
    looking for benzyl deprotection followed by immediate re-protection 
    or selectivity issues due to unprotected phenol.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "phenol")
        self.protecting_group = config.get("protecting_group", "benzyl")
        self.cycle_present = config.get("cycle_present", True)
        self.consecutive_steps = config.get("consecutive_steps", True)
        
        # SMARTS patterns for detection
        self.benzyl_protected_phenol = "[OH0:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[OH0:8][CH2:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1"
        self.free_phenol = "[OH1:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
        self.benzyl_group = "[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track deprotection and protection events
        deprotection_steps = []
        protection_steps = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_benzyl_deprotection(rxn):
                deprotection_steps.append(i)
            elif self.detect_benzyl_protection(rxn):
                protection_steps.append(i)
        
        # Check for protection cycle
        cycle_detected = self.detect_protection_cycle(deprotection_steps, protection_steps)
        
        condition = cycle_detected == self.cycle_present
        return condition, len(reactions)
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect removal of benzyl protecting group from phenol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check if reactant has benzyl-protected phenol
            reactant_mol = Chem.MolFromSmiles(reactants)
            if not reactant_mol:
                return False
                
            # Look for benzyl ether pattern being broken
            benzyl_ether_pattern = Chem.MolFromSmarts("[OH0]([c]1ccccc1)[CH2][c]1ccccc1")
            if not reactant_mol.HasSubstructMatch(benzyl_ether_pattern):
                return False
            
            # Check if product has free phenol
            product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
            free_phenol_pattern = Chem.MolFromSmarts("[OH1][c]1ccccc1")
            benzyl_fragment = Chem.MolFromSmarts("[CH2][c]1ccccc1")
            
            has_free_phenol = any(mol and mol.HasSubstructMatch(free_phenol_pattern) for mol in product_mols)
            has_benzyl_fragment = any(mol and mol.HasSubstructMatch(benzyl_fragment) for mol in product_mols)
            
            return has_free_phenol and has_benzyl_fragment
            
        except:
            return False
    
    def detect_benzyl_protection(self, rxn):
        """Detect formation of benzyl protecting group on phenol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check if reactant has free phenol
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            free_phenol_pattern = Chem.MolFromSmarts("[OH1][c]1ccccc1")
            benzyl_reagent_pattern = Chem.MolFromSmarts("[CH2][c]1ccccc1")
            
            has_free_phenol = any(mol and mol.HasSubstructMatch(free_phenol_pattern) for mol in reactant_mols)
            has_benzyl_reagent = any(mol and mol.HasSubstructMatch(benzyl_reagent_pattern) for mol in reactant_mols)
            
            if not (has_free_phenol and has_benzyl_reagent):
                return False
            
            # Check if product has benzyl-protected phenol
            product_mol = Chem.MolFromSmiles(products)
            if not product_mol:
                return False
                
            benzyl_ether_pattern = Chem.MolFromSmarts("[OH0]([c]1ccccc1)[CH2][c]1ccccc1")
            return product_mol.HasSubstructMatch(benzyl_ether_pattern)
            
        except:
            return False
    
    def detect_protection_cycle(self, deprotection_steps, protection_steps):
        """Detect if deprotection is followed by re-protection"""
        if not deprotection_steps:
            return False
            
        # Check for consecutive deprotection-protection
        if self.consecutive_steps:
            for dep_step in deprotection_steps:
                # Look for protection in the next few steps
                nearby_protection = any(
                    prot_step > dep_step and prot_step <= dep_step + 3 
                    for prot_step in protection_steps
                )
                if nearby_protection:
                    return True
        else:
            # Just check if both deprotection and protection occur
            return len(deprotection_steps) > 0 and len(protection_steps) > 0
            
        return False
