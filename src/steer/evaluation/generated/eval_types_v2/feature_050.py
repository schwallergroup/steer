"""Generated evaluation code for: Mixed ester protecting group approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MixedEsterProtectingGroup(MultiRxnCondBase):
    """
    Evaluates routes that employ mixed ester protecting group strategies with 
    tert-butyl and ethyl esters for differential deprotection.
    """
    
    def __init__(self, config):
        self.require_tert_butyl = config.get("require_tert_butyl", True)
        self.require_ethyl = config.get("require_ethyl", True)
        self.require_differential = config.get("require_differential", True)
        
        # SMARTS patterns for ester protecting groups
        self.tert_butyl_ester = "C(=O)OC(C)(C)C"
        self.ethyl_ester = "C(=O)OCC"
        
        # Patterns for deprotection reactions (ester hydrolysis)
        self.ester_hydrolysis = "C(=O)O[C,c]"
    
    def condition_depth(self, d):
        """Check if the route uses mixed ester protecting group strategy"""
        reactions = self.get_rxns(d)
        
        has_tert_butyl_protection = False
        has_ethyl_protection = False
        has_tert_butyl_deprotection = False
        has_ethyl_deprotection = False
        
        for rxn in reactions:
            # Check for protection reactions (formation of esters)
            if self.detect_ester_protection(rxn, self.tert_butyl_ester):
                has_tert_butyl_protection = True
            if self.detect_ester_protection(rxn, self.ethyl_ester):
                has_ethyl_protection = True
                
            # Check for deprotection reactions (hydrolysis of esters)
            if self.detect_ester_deprotection(rxn, self.tert_butyl_ester):
                has_tert_butyl_deprotection = True
            if self.detect_ester_deprotection(rxn, self.ethyl_ester):
                has_ethyl_deprotection = True
        
        # Evaluate strategy conditions
        tert_butyl_condition = (not self.require_tert_butyl) or \
                              (has_tert_butyl_protection and has_tert_butyl_deprotection)
        ethyl_condition = (not self.require_ethyl) or \
                         (has_ethyl_protection and has_ethyl_deprotection)
        
        # For differential strategy, both protecting groups should be used
        differential_condition = (not self.require_differential) or \
                               (has_tert_butyl_protection and has_ethyl_protection)
        
        condition_met = tert_butyl_condition and ethyl_condition and differential_condition
        
        return condition_met, len(reactions)
    
    def detect_ester_protection(self, rxn, ester_pattern):
        """Detect ester protection (carboxylic acid -> ester)"""
        try:
            rxn_parts = rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if carboxylic acid in reactants and ester in products
            carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")
            ester_mol_pattern = Chem.MolFromSmarts(ester_pattern)
            
            has_carboxylic_acid_reactant = any(mol.HasSubstructMatch(carboxylic_acid_pattern) 
                                             for mol in reactants)
            has_ester_product = any(mol.HasSubstructMatch(ester_mol_pattern) 
                                  for mol in products)
            
            return has_carboxylic_acid_reactant and has_ester_product
            
        except Exception:
            return False
    
    def detect_ester_deprotection(self, rxn, ester_pattern):
        """Detect ester deprotection (ester -> carboxylic acid)"""
        try:
            rxn_parts = rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if specific ester in reactants and carboxylic acid in products
            ester_mol_pattern = Chem.MolFromSmarts(ester_pattern)
            carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")
            
            has_ester_reactant = any(mol.HasSubstructMatch(ester_mol_pattern) 
                                   for mol in reactants)
            has_carboxylic_acid_product = any(mol.HasSubstructMatch(carboxylic_acid_pattern) 
                                            for mol in products)
            
            return has_ester_reactant and has_carboxylic_acid_product
            
        except Exception:
            return False
