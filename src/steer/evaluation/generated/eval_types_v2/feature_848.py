"""Generated evaluation code for: Boronic ester protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BoronicEsterProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates the use of boronic ester protecting group strategy in synthesis routes.
    Checks for the presence of both protection (boronic acid -> boronic ester) and 
    deprotection (boronic ester -> boronic acid) reactions when required.
    """
    
    def __init__(self, config):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.deprotection_present = config["parameters"]["deprotection_present"]
        
        # SMARTS patterns for boronic acid and boronic ester
        self.boronic_acid_pattern = "[B](O)(O)"
        self.boronic_ester_pattern = "[B]1OCC(C)(C)CO1"  # Pinacol boronate ester
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = False
        deprotection_found = False
        
        for rxn in reactions:
            # Check for protection: boronic acid -> boronic ester
            if self.detect_protection(rxn):
                protection_found = True
                
            # Check for deprotection: boronic ester -> boronic acid
            if self.detect_deprotection(rxn):
                deprotection_found = True
        
        # Strategy is complete if we find protection and (if required) deprotection
        if self.deprotection_present:
            condition = protection_found and deprotection_found
        else:
            condition = protection_found
            
        return condition, len(reactions)
    
    def detect_protection(self, rxn):
        """
        Detects boronic acid to boronic ester protection reaction.
        Reactants should contain boronic acid, products should contain boronic ester.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if any reactant has boronic acid
            reactant_has_boronic_acid = False
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boronic_acid_pattern)):
                    reactant_has_boronic_acid = True
                    break
            
            # Check if any product has boronic ester
            product_has_boronic_ester = False
            for p_smiles in products:
                mol = Chem.MolFromSmiles(p_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boronic_ester_pattern)):
                    product_has_boronic_ester = True
                    break
            
            return reactant_has_boronic_acid and product_has_boronic_ester
            
        except Exception:
            return False
    
    def detect_deprotection(self, rxn):
        """
        Detects boronic ester to boronic acid deprotection reaction.
        Reactants should contain boronic ester, products should contain boronic acid.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if any reactant has boronic ester
            reactant_has_boronic_ester = False
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boronic_ester_pattern)):
                    reactant_has_boronic_ester = True
                    break
            
            # Check if any product has boronic acid
            product_has_boronic_acid = False
            for p_smiles in products:
                mol = Chem.MolFromSmiles(p_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boronic_acid_pattern)):
                    product_has_boronic_acid = True
                    break
            
            return reactant_has_boronic_ester and product_has_boronic_acid
            
        except Exception:
            return False
