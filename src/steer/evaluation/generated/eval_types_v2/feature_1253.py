"""Generated evaluation code for: TMSE protecting group strategy for alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TMSEProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates TMSE (2-(trimethylsilyl)ethoxy) protecting group strategy for alcohols.
    Checks for presence of TMSE protection and appropriate deprotection timing.
    """
    
    def __init__(self, config):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.deprotection_timing = config["parameters"]["deprotection_timing"]
        
        # TMSE protection pattern: -O-CH2-CH2-Si(CH3)3
        self.tmse_pattern = "O[CH2][CH2][Si]([CH3])([CH3])[CH3]"
        # Alcohol pattern
        self.alcohol_pattern = "[OH]"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_protection = False
        has_deprotection = False
        protection_depth = -1
        deprotection_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_tmse_protection(rxn):
                has_protection = True
                protection_depth = i
            elif self.detect_tmse_deprotection(rxn):
                has_deprotection = True
                deprotection_depth = i
        
        # Check if strategy is properly implemented
        strategy_complete = has_protection and has_deprotection
        
        # Check deprotection timing (final = last few steps)
        proper_timing = True
        if has_deprotection and self.deprotection_timing == "final":
            total_steps = len(reactions)
            # Consider final 20% of steps as "final"
            final_threshold = max(1, int(0.8 * total_steps))
            proper_timing = deprotection_depth >= final_threshold
        
        condition_met = strategy_complete and proper_timing
        
        # Return average depth of protection/deprotection events
        if has_protection and has_deprotection:
            avg_depth = (protection_depth + deprotection_depth) / 2
            return condition_met, int(avg_depth)
        elif has_protection:
            return condition_met, protection_depth
        else:
            return condition_met, len(reactions)
    
    def detect_tmse_protection(self, rxn):
        """Detect TMSE protection reaction (alcohol -> TMSE ether)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[0].split(".") if s.strip()]
        products = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[1].split(".") if s.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check if reactants contain alcohol and products contain TMSE ether
        has_alcohol_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.alcohol_pattern)) 
                                 for mol in reactants if mol)
        has_tmse_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.tmse_pattern)) 
                              for mol in products if mol)
        
        # Should have alcohol in reactants but not TMSE, and TMSE in products
        has_tmse_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.tmse_pattern)) 
                               for mol in reactants if mol)
        
        return has_alcohol_reactant and has_tmse_product and not has_tmse_reactant
    
    def detect_tmse_deprotection(self, rxn):
        """Detect TMSE deprotection reaction (TMSE ether -> alcohol)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[0].split(".") if s.strip()]
        products = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[1].split(".") if s.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check if reactants contain TMSE ether and products contain alcohol
        has_tmse_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.tmse_pattern)) 
                               for mol in reactants if mol)
        has_alcohol_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.alcohol_pattern)) 
                                 for mol in products if mol)
        
        # Should have TMSE in reactants but not products, and alcohol in products
        has_tmse_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.tmse_pattern)) 
                              for mol in products if mol)
        
        return has_tmse_reactant and has_alcohol_product and not has_tmse_product
