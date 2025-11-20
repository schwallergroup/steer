"""Generated evaluation code for: TBDMS protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBDMSProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates TBDMS protecting group strategy for alcohols.
    Checks if TBDMS protection is applied to alcohols and later removed appropriately.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "TBDMS")
        self.functional_group = config.get("functional_group", "alcohol")
        self.target_depth = config.get("target_depth", {"type": "bool", "value": True})
        
        # TBDMS protection patterns
        self.tbdms_pattern = Chem.MolFromSmarts("[Si](C)(C)(C(C)(C)C)[OH1,OH0]")
        self.alcohol_pattern = Chem.MolFromSmarts("[OH1][C]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if TBDMS protecting group strategy is properly employed"""
        reactions = self.get_rxns(d)
        
        has_protection = False
        has_deprotection = False
        protection_depth = -1
        deprotection_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_tbdms_protection(rxn):
                has_protection = True
                protection_depth = i
            elif self.detect_tbdms_deprotection(rxn):
                has_deprotection = True
                deprotection_depth = i
                
        # Strategy is successful if both protection and deprotection occur
        # and protection happens before deprotection (lower depth number)
        condition = (has_protection and has_deprotection and 
                    protection_depth < deprotection_depth)
        
        if condition:
            # Return the depth where deprotection occurs as the key step
            return condition, deprotection_depth + 1
        else:
            return condition, len(reactions)
    
    def detect_tbdms_protection(self, rxn):
        """Detect TBDMS protection reaction (alcohol -> TBDMS ether)"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Check if reactants have free alcohol and products have TBDMS
            has_reactant_alcohol = any(
                mol.HasSubstructMatch(self.alcohol_pattern) for mol in reactants if mol
            )
            has_product_tbdms = any(
                mol.HasSubstructMatch(self.tbdms_pattern) for mol in products if mol
            )
            
            return has_reactant_alcohol and has_product_tbdms
            
        except:
            return False
    
    def detect_tbdms_deprotection(self, rxn):
        """Detect TBDMS deprotection reaction (TBDMS ether -> alcohol)"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Check if reactants have TBDMS and products have free alcohol
            has_reactant_tbdms = any(
                mol.HasSubstructMatch(self.tbdms_pattern) for mol in reactants if mol
            )
            has_product_alcohol = any(
                mol.HasSubstructMatch(self.alcohol_pattern) for mol in products if mol
            )
            
            return has_reactant_tbdms and has_product_alcohol
            
        except:
            return False
    
    def route_scoring(self, x) -> float:
        """Convert strategy success to score"""
        if self.target_depth["type"] == "bool":
            if self.target_depth["value"]:
                return 1.0 if x >= 0 else 0.0  # Reward successful strategy
            else:
                return 0.0 if x >= 0 else 1.0  # Penalize strategy use
        else:
            # For non-boolean targets, could implement depth-based scoring
            return 1.0 if x >= 0 else 0.0
