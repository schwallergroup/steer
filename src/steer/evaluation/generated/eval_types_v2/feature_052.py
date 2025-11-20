"""Generated evaluation code for: Tertiary alcohol acetate protection-deprotection cycle"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TertiaryAlcoholAcetateProtectionCycle(MultiRxnCondBase):
    """
    Evaluates routes for tertiary alcohol acetate protection-deprotection cycles.
    Checks if a tertiary alcohol is protected as an acetate and later deprotected.
    """
    
    def __init__(self, config):
        self.require_cycle = config.get("require_cycle", True)
        self.tertiary_alcohol_pattern = "[CH0](C)(C)(C)O"  # Tertiary alcohol pattern
        self.acetate_pattern = "[CH0](C)(C)(C)OC(=O)C"     # Tertiary acetate pattern
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_found = False
        deprotection_found = False
        
        for rxn in reactions:
            if self.detect_acetate_protection(rxn):
                protection_found = True
            if self.detect_acetate_deprotection(rxn):
                deprotection_found = True
        
        if self.require_cycle:
            condition = protection_found and deprotection_found
        else:
            condition = protection_found or deprotection_found
            
        return condition, len(reactions)
    
    def detect_acetate_protection(self, rxn):
        """
        Detects tertiary alcohol -> tertiary acetate transformation
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check if reactant has tertiary alcohol and product has tertiary acetate
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for tertiary alcohol in reactants
            has_tert_alcohol_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.tertiary_alcohol_pattern))
                for mol in reactant_mols
            )
            
            # Check for tertiary acetate in products
            has_tert_acetate_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.acetate_pattern))
                for mol in product_mols
            )
            
            return has_tert_alcohol_reactant and has_tert_acetate_product
            
        except:
            return False
    
    def detect_acetate_deprotection(self, rxn):
        """
        Detects tertiary acetate -> tertiary alcohol transformation
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check if reactant has tertiary acetate and product has tertiary alcohol
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for tertiary acetate in reactants
            has_tert_acetate_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.acetate_pattern))
                for mol in reactant_mols
            )
            
            # Check for tertiary alcohol in products
            has_tert_alcohol_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.tertiary_alcohol_pattern))
                for mol in product_mols
            )
            
            return has_tert_acetate_reactant and has_tert_alcohol_product
            
        except:
            return False
