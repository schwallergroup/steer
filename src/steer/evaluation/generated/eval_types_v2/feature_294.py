"""Generated evaluation code for: Carboxylic acid protection-deprotection cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CarboxylicAcidProtectionCycle(MultiRxnCondBase):
    """
    Checks for carboxylic acid protection-deprotection cycling using ethyl ester.
    Detects if carboxylic acid is protected as ethyl ester and later deprotected.
    """
    
    def __init__(self, config):
        self.functional_group = config["parameters"]["functional_group"]
        self.protecting_group = config["parameters"]["protecting_group"]
        self.cycle_count = config["parameters"]["cycle_count"]
        
        # SMARTS patterns
        self.carboxylic_acid_pattern = "[C](=[O])[OH]"
        self.ethyl_ester_pattern = "[C](=[O])[O][CH2][CH3]"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = False
        deprotection_found = False
        
        for rxn in reactions:
            if self.detect_protection(rxn):
                protection_found = True
            if self.detect_deprotection(rxn):
                deprotection_found = True
        
        # Check if we have the complete cycle
        condition = protection_found and deprotection_found
        return condition, len(reactions)
    
    def detect_protection(self, rxn):
        """Detect carboxylic acid -> ethyl ester protection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if reactant has carboxylic acid and product has ethyl ester
        reactant_has_acid = False
        product_has_ester = False
        
        for r_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern)):
                    reactant_has_acid = True
                    break
            except:
                continue
        
        for p_smiles in products:
            try:
                mol = Chem.MolFromSmiles(p_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.ethyl_ester_pattern)):
                    product_has_ester = True
                    break
            except:
                continue
        
        return reactant_has_acid and product_has_ester
    
    def detect_deprotection(self, rxn):
        """Detect ethyl ester -> carboxylic acid deprotection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if reactant has ethyl ester and product has carboxylic acid
        reactant_has_ester = False
        product_has_acid = False
        
        for r_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.ethyl_ester_pattern)):
                    reactant_has_ester = True
                    break
            except:
                continue
        
        for p_smiles in products:
            try:
                mol = Chem.MolFromSmiles(p_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern)):
                    product_has_acid = True
                    break
            except:
                continue
        
        return reactant_has_ester and product_has_acid
