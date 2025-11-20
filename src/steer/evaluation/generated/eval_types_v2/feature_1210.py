"""Generated evaluation code for: Benzyl protecting group strategy for alcohol differentiation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes for benzyl protecting group strategy on alcohols.
    Checks if benzyl ethers are formed for alcohol protection and later removed,
    indicating a protecting group differentiation strategy.
    """
    
    def __init__(self, config):
        self.require_protection = config.get("require_protection", True)
        self.require_deprotection = config.get("require_deprotection", True)
        self.min_steps_between = config.get("min_steps_between", 1)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = False
        deprotection_found = False
        protection_step = -1
        deprotection_step = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_benzyl_protection(rxn):
                protection_found = True
                protection_step = i
            elif self.detect_benzyl_deprotection(rxn):
                deprotection_found = True
                deprotection_step = i
        
        # Check if strategy is properly implemented
        strategy_condition = True
        
        if self.require_protection and not protection_found:
            strategy_condition = False
        
        if self.require_deprotection and not deprotection_found:
            strategy_condition = False
            
        # Check if protection comes before deprotection with minimum steps
        if protection_found and deprotection_found:
            if protection_step >= deprotection_step:
                strategy_condition = False
            elif (deprotection_step - protection_step) < self.min_steps_between:
                strategy_condition = False
        
        return strategy_condition, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """Detect formation of benzyl ether from alcohol"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Look for alcohol in reactants and benzyl ether in products
            alcohol_pattern = Chem.MolFromSmarts("[OH1]")  # Primary or secondary alcohol
            benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1CO")  # Benzyl ether
            
            has_alcohol_reactant = False
            has_benzyl_product = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(alcohol_pattern):
                    has_alcohol_reactant = True
                    break
            
            for product_smiles in products:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(benzyl_ether_pattern):
                    has_benzyl_product = True
                    break
            
            return has_alcohol_reactant and has_benzyl_product
            
        except:
            return False
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect removal of benzyl protecting group to regenerate alcohol"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Look for benzyl ether in reactants and alcohol in products
            benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1CO")  # Benzyl ether
            alcohol_pattern = Chem.MolFromSmarts("[OH1]")  # Alcohol
            
            has_benzyl_reactant = False
            has_alcohol_product = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(benzyl_ether_pattern):
                    has_benzyl_reactant = True
                    break
            
            for product_smiles in products:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(alcohol_pattern):
                    has_alcohol_product = True
                    break
            
            return has_benzyl_reactant and has_alcohol_product
            
        except:
            return False
