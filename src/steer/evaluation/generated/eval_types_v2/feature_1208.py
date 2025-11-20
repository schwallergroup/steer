"""Generated evaluation code for: Benzyl protection deprotection cycle"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectionCycle(MultiRxnCondBase):
    """
    Checks for benzyl protection-deprotection cycle of alcohols.
    Detects both benzyl protection of alcohols and subsequent deprotection.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "benzyl")
        self.functional_group = config.get("functional_group", "alcohol")
        self.has_cycle = config.get("has_cycle", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = any(self.detect_benzyl_protection(r) for r in reactions)
        deprotection_found = any(self.detect_benzyl_deprotection(r) for r in reactions)
        
        if self.has_cycle:
            # Both protection and deprotection must be present
            condition = protection_found and deprotection_found
        else:
            # Only protection or only deprotection
            condition = protection_found or deprotection_found
            
        return condition, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection of alcohols (R-OH + BnX -> R-O-Bn)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check for alcohol in reactants and benzyl ether in products
        alcohol_pattern = "[OH1][CH,CH2,CH3]"  # Primary/secondary/tertiary alcohol
        benzyl_ether_pattern = "[CH2][c]1[cH][cH][cH][cH][cH]1-[O][CH,CH2,CH3]"  # Benzyl ether
        
        has_alcohol_reactant = False
        has_benzyl_ether_product = False
        
        for reactant in reactants:
            mol = Chem.MolFromSmiles(reactant)
            if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(alcohol_pattern)):
                has_alcohol_reactant = True
                
        for product in products:
            mol = Chem.MolFromSmiles(product)
            if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(benzyl_ether_pattern)):
                has_benzyl_ether_product = True
                
        return has_alcohol_reactant and has_benzyl_ether_product
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl deprotection (R-O-Bn -> R-OH)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check for benzyl ether in reactants and alcohol in products
        benzyl_ether_pattern = "[CH2][c]1[cH][cH][cH][cH][cH]1-[O][CH,CH2,CH3]"  # Benzyl ether
        alcohol_pattern = "[OH1][CH,CH2,CH3]"  # Primary/secondary/tertiary alcohol
        
        has_benzyl_ether_reactant = False
        has_alcohol_product = False
        
        for reactant in reactants:
            mol = Chem.MolFromSmiles(reactant)
            if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(benzyl_ether_pattern)):
                has_benzyl_ether_reactant = True
                
        for product in products:
            mol = Chem.MolFromSmiles(product)
            if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(alcohol_pattern)):
                has_alcohol_product = True
                
        return has_benzyl_ether_reactant and has_alcohol_product
