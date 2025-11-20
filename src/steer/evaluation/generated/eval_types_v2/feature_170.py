"""Generated evaluation code for: Dual Boc protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualBocCyclingStrategy(MultiRxnCondBase):
    """
    Evaluates routes that use a dual Boc protecting group cycling strategy.
    Checks for multiple Boc protection/deprotection cycles throughout the synthesis.
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("min_cycles", 2)
        self.require_selective = config.get("require_selective", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        boc_protection_count = 0
        boc_deprotection_count = 0
        selective_removal_attempted = False
        
        for rxn in reactions:
            if self.detect_boc_protection(rxn):
                boc_protection_count += 1
            elif self.detect_boc_deprotection(rxn):
                boc_deprotection_count += 1
                
            if self.detect_selective_boc_removal(rxn):
                selective_removal_attempted = True
        
        # Calculate number of complete cycles (protection followed by deprotection)
        cycles = min(boc_protection_count, boc_deprotection_count)
        
        # Check if cycling strategy criteria are met
        has_multiple_cycles = cycles >= self.min_cycles
        has_selective_removal = not self.require_selective or selective_removal_attempted
        
        condition = has_multiple_cycles and has_selective_removal
        
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection reaction (amine -> Boc-protected amine)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Look for Boc anhydride or Boc-Cl as reagent
        boc_reagents = [
            "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O
            "CC(C)(C)OC(=O)Cl"  # Boc-Cl
        ]
        
        has_boc_reagent = any(reagent in reactants for reagent in boc_reagents)
        
        # Check for increase in Boc groups from reactants to products
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".") if smi.strip()]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".") if smi.strip()]
        
        boc_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")
        
        reactant_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in reactant_mols if mol)
        product_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in product_mols if mol)
        
        return has_boc_reagent and product_boc_count > reactant_boc_count
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection reaction (Boc-protected amine -> amine)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Look for common Boc deprotection conditions (TFA, HCl)
        deprotection_reagents = [
            "C(=O)(C(F)(F)F)O",  # TFA
            "Cl",  # HCl
            "ClC(Cl)Cl"  # DCM (sometimes used with TFA)
        ]
        
        has_deprotection_conditions = any(reagent in reactants for reagent in deprotection_reagents)
        
        # Check for decrease in Boc groups from reactants to products
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".") if smi.strip()]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".") if smi.strip()]
        
        boc_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")
        
        reactant_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in reactant_mols if mol)
        product_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in product_mols if mol)
        
        return has_deprotection_conditions and reactant_boc_count > product_boc_count
    
    def detect_selective_boc_removal(self, rxn):
        """Detect selective Boc removal where only some Boc groups are removed"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".") if smi.strip()]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".") if smi.strip()]
        
        boc_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")
        
        reactant_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in reactant_mols if mol)
        product_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in product_mols if mol)
        
        # Selective removal: some but not all Boc groups removed
        return reactant_boc_count > product_boc_count > 0 and product_boc_count >= 1
