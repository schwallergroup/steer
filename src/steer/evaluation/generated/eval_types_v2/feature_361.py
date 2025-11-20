"""Generated evaluation code for: Protect deprotect acetate strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectDeprotectAcetateStrategy(MultiRxnCondBase):
    """
    Evaluates routes for acetate protection/deprotection strategy on primary alcohols.
    Checks if the route contains both acetate protection and deprotection reactions
    forming a complete protect-deprotect cycle.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "acetate")
        self.functional_group = config.get("functional_group", "primary_alcohol")
        self.cycle_present = config.get("cycle_present", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for acetate protection (alcohol -> acetate ester)
        protection_found = any(self.detect_acetate_protection(r) for r in reactions)
        
        # Check for acetate deprotection (acetate ester -> alcohol)
        deprotection_found = any(self.detect_acetate_deprotection(r) for r in reactions)
        
        # Strategy requires both protection and deprotection if cycle_present is True
        if self.cycle_present:
            condition = protection_found and deprotection_found
        else:
            condition = protection_found or deprotection_found
            
        return condition, len(reactions)
    
    def detect_acetate_protection(self, rxn):
        """Detect acetate protection: primary alcohol + acetyl source -> acetate ester"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Primary alcohol pattern
            primary_alcohol_pattern = Chem.MolFromSmarts("[CH2][OH]")
            # Acetate ester pattern  
            acetate_ester_pattern = Chem.MolFromSmarts("[CH2]OC(=O)[CH3]")
            # Acetyl source patterns (acetic anhydride, acetyl chloride, etc.)
            acetyl_patterns = [
                Chem.MolFromSmarts("CC(=O)OC(=O)C"),  # acetic anhydride
                Chem.MolFromSmarts("CC(=O)Cl"),        # acetyl chloride
                Chem.MolFromSmarts("CC(=O)O")          # acetic acid
            ]
            
            # Check if reactants contain primary alcohol and acetyl source
            has_primary_alcohol = False
            has_acetyl_source = False
            
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol is None:
                    continue
                    
                if primary_alcohol_pattern.HasSubstructMatch(reactant_mol):
                    has_primary_alcohol = True
                    
                for acetyl_pattern in acetyl_patterns:
                    if acetyl_pattern.HasSubstructMatch(reactant_mol):
                        has_acetyl_source = True
                        break
            
            # Check if products contain acetate ester
            has_acetate_ester = False
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol is None:
                    continue
                    
                if acetate_ester_pattern.HasSubstructMatch(product_mol):
                    has_acetate_ester = True
                    break
            
            return has_primary_alcohol and has_acetyl_source and has_acetate_ester
            
        except:
            return False
    
    def detect_acetate_deprotection(self, rxn):
        """Detect acetate deprotection: acetate ester -> primary alcohol"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Acetate ester pattern
            acetate_ester_pattern = Chem.MolFromSmarts("[CH2]OC(=O)[CH3]")
            # Primary alcohol pattern
            primary_alcohol_pattern = Chem.MolFromSmarts("[CH2][OH]")
            
            # Check if reactants contain acetate ester
            has_acetate_ester = False
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol is None:
                    continue
                    
                if acetate_ester_pattern.HasSubstructMatch(reactant_mol):
                    has_acetate_ester = True
                    break
            
            # Check if products contain primary alcohol
            has_primary_alcohol = False
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol is None:
                    continue
                    
                if primary_alcohol_pattern.HasSubstructMatch(product_mol):
                    has_primary_alcohol = True
                    break
            
            return has_acetate_ester and has_primary_alcohol
            
        except:
            return False
