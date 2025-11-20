"""Generated evaluation code for: Benzyl protecting group deprotection sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates benzyl protecting group deprotection sequences.
    Checks for benzyl protection at specified steps, followed by deprotection,
    with intermediate transformations maintaining the protecting group.
    """
    
    def __init__(self, config):
        self.steps_with_protection = set(config["parameters"]["steps_with_protection"])
        self.steps_with_deprotection = set(config["parameters"]["steps_with_deprotection"])
        self.intermediate_steps = set(config["parameters"]["intermediate_steps"])
        
        # Benzyl protecting group patterns
        self.benzyl_ether_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1-O-[#6]")  # Bn-O-R
        self.benzyl_ester_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1-O-C(=O)-[#6]")  # Bn-O-CO-R
        self.benzyl_amine_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1-N-[#6]")  # Bn-N-R
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_steps = len(reactions)
        
        # Check each step according to requirements
        protection_found = False
        deprotection_found = False
        intermediate_maintained = True
        
        for i, rxn in enumerate(reactions):
            step_num = i + 1
            
            if step_num in self.steps_with_protection:
                protection_found = self.detect_benzyl_protection(rxn)
                
            elif step_num in self.steps_with_deprotection:
                deprotection_found = self.detect_benzyl_deprotection(rxn)
                
            elif step_num in self.intermediate_steps:
                # Benzyl group should be present but not modified
                if not self.benzyl_group_maintained(rxn):
                    intermediate_maintained = False
        
        condition = protection_found and deprotection_found and intermediate_maintained
        return condition, total_steps
    
    def detect_benzyl_protection(self, rxn):
        """Detect formation of benzyl protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check if benzyl group is absent in reactants but present in products
        reactant_has_benzyl = any(self.has_benzyl_protecting_group(mol) for mol in reactants if mol)
        product_has_benzyl = any(self.has_benzyl_protecting_group(mol) for mol in products if mol)
        
        return not reactant_has_benzyl and product_has_benzyl
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect removal of benzyl protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check if benzyl group is present in reactants but absent in products
        reactant_has_benzyl = any(self.has_benzyl_protecting_group(mol) for mol in reactants if mol)
        product_has_benzyl = any(self.has_benzyl_protecting_group(mol) for mol in products if mol)
        
        # Also check for benzyl alcohol or toluene as byproducts
        benzyl_byproduct = any(self.is_benzyl_byproduct(mol) for mol in products if mol)
        
        return reactant_has_benzyl and not product_has_benzyl and benzyl_byproduct
    
    def benzyl_group_maintained(self, rxn):
        """Check if benzyl protecting group is maintained through the reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        reactant_has_benzyl = any(self.has_benzyl_protecting_group(mol) for mol in reactants if mol)
        product_has_benzyl = any(self.has_benzyl_protecting_group(mol) for mol in products if mol)
        
        return reactant_has_benzyl and product_has_benzyl
    
    def has_benzyl_protecting_group(self, mol):
        """Check if molecule contains benzyl protecting group"""
        if not mol:
            return False
            
        return (mol.HasSubstructMatch(self.benzyl_ether_pattern) or
                mol.HasSubstructMatch(self.benzyl_ester_pattern) or
                mol.HasSubstructMatch(self.benzyl_amine_pattern))
    
    def is_benzyl_byproduct(self, mol):
        """Check if molecule is a benzyl deprotection byproduct"""
        if not mol:
            return False
            
        # Common benzyl deprotection byproducts
        benzyl_alcohol = Chem.MolFromSmarts("OCc1ccccc1")
        toluene = Chem.MolFromSmarts("Cc1ccccc1")
        
        return (mol.HasSubstructMatch(benzyl_alcohol) or 
                mol.HasSubstructMatch(toluene))
