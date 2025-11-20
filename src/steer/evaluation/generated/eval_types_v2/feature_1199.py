"""Generated evaluation code for: Acid protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcidProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on acid protecting group cycling strategy.
    Checks for multiple protection-deprotection cycles of carboxylic acids through ester forms.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "carboxylic_acid")
        self.required_protection_count = config.get("protection_count", 2)
        self.required_deprotection_count = config.get("deprotection_count", 2)
        self.strategy_type = config.get("strategy_type", "cycling")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_count = 0
        deprotection_count = 0
        
        for rxn in reactions:
            if self.detect_acid_protection(rxn):
                protection_count += 1
            elif self.detect_acid_deprotection(rxn):
                deprotection_count += 1
        
        # Check if cycling strategy is met
        protection_met = protection_count >= self.required_protection_count
        deprotection_met = deprotection_count >= self.required_deprotection_count
        cycling_condition = protection_met and deprotection_met
        
        return cycling_condition, len(reactions)
    
    def detect_acid_protection(self, rxn):
        """Detect carboxylic acid to ester protection reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # SMARTS pattern for carboxylic acid
        acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        # SMARTS pattern for ester
        ester_pattern = Chem.MolFromSmarts("[CX3](=O)[O][C,c]")
        
        # Check if we have acid in reactants and ester in products
        acid_in_reactants = False
        ester_in_products = False
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(acid_pattern):
                    acid_in_reactants = True
                    break
            except:
                continue
        
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(ester_pattern):
                    ester_in_products = True
                    break
            except:
                continue
        
        return acid_in_reactants and ester_in_products
    
    def detect_acid_deprotection(self, rxn):
        """Detect ester to carboxylic acid deprotection reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # SMARTS pattern for ester
        ester_pattern = Chem.MolFromSmarts("[CX3](=O)[O][C,c]")
        # SMARTS pattern for carboxylic acid
        acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        
        # Check if we have ester in reactants and acid in products
        ester_in_reactants = False
        acid_in_products = False
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(ester_pattern):
                    ester_in_reactants = True
                    break
            except:
                continue
        
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(acid_pattern):
                    acid_in_products = True
                    break
            except:
                continue
        
        return ester_in_reactants and acid_in_products
