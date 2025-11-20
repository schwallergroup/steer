"""Generated evaluation code for: Benzyl protecting group strategy for alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates benzyl protecting group strategy for alcohols.
    Checks if alcohol is protected as benzyl ether and remains protected 
    for the specified number of steps before deprotection.
    """
    
    def __init__(self, config):
        self.protecting_group = config["protecting_group"]  # "benzyl"
        self.functional_group = config["functional_group"]  # "alcohol" 
        self.steps_protected = config["steps_protected"]  # 4
        
        # SMARTS patterns
        self.benzyl_ether_pattern = "[OH0]([CH2][cR6])[CH,CH2,CH3]"  # R-O-CH2-Ph
        self.free_alcohol_pattern = "[OH1][CH,CH2,CH3]"  # R-OH
        self.benzyl_protection_rxn = "[OH1:1][CH,CH2,CH3:2]>>[OH0:1]([CH2][cR6])[CH,CH2,CH3:2]"
        self.benzyl_deprotection_rxn = "[OH0:1]([CH2][cR6])[CH,CH2,CH3:2]>>[OH1:1][CH,CH2,CH3:2]"

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find protection and deprotection reactions
        protection_step = -1
        deprotection_step = -1
        
        for i, rxn in enumerate(reactions):
            if self.is_benzyl_protection(rxn):
                protection_step = i
            elif self.is_benzyl_deprotection(rxn):
                deprotection_step = i
                
        # Check if strategy is properly implemented
        if protection_step == -1 or deprotection_step == -1:
            return False, len(reactions)
            
        # Protection should occur before deprotection
        if protection_step >= deprotection_step:
            return False, len(reactions)
            
        # Check if protected for required number of steps
        steps_protected = deprotection_step - protection_step
        condition_met = steps_protected >= self.steps_protected
        
        return condition_met, len(reactions)

    def is_benzyl_protection(self, rxn):
        """Check if reaction converts alcohol to benzyl ether"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check if reactants contain free alcohol
            has_free_alcohol = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_alcohol_pattern))
                for mol in reactant_mols
            )
            
            # Check if products contain benzyl ether
            has_benzyl_ether = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern))
                for mol in product_mols
            )
            
            return has_free_alcohol and has_benzyl_ether
            
        except:
            return False

    def is_benzyl_deprotection(self, rxn):
        """Check if reaction converts benzyl ether back to alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check if reactants contain benzyl ether
            has_benzyl_ether = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern))
                for mol in reactant_mols
            )
            
            # Check if products contain free alcohol
            has_free_alcohol = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_alcohol_pattern))
                for mol in product_mols
            )
            
            return has_benzyl_ether and has_free_alcohol
            
        except:
            return False

    def route_scoring(self, x):
        """Score based on whether protecting group strategy is implemented"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 10  # Strategy successfully implemented
