"""Generated evaluation code for: Protecting group exchange strategy TBDMS to benzyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupExchange(MultiRxnCondBase):
    """
    Evaluates synthesis routes for protecting group exchange strategies,
    specifically TBDMS to benzyl ether conversion on phenolic groups.
    Checks for sequential deprotection followed by reprotection steps.
    """
    
    def __init__(self, config):
        self.exchange_type = config.get("exchange_type", "silyl_to_benzyl")
        self.functional_group = config.get("functional_group", "phenol")
        self.sequential_steps = config.get("sequential_steps", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find TBDMS deprotection and benzyl protection reactions
        tbdms_deprotection = []
        benzyl_protection = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_tbdms_deprotection(rxn):
                tbdms_deprotection.append(i)
            if self.detect_benzyl_protection(rxn):
                benzyl_protection.append(i)
        
        # Check if we have the protecting group exchange
        has_exchange = len(tbdms_deprotection) > 0 and len(benzyl_protection) > 0
        
        if self.sequential_steps and has_exchange:
            # Verify that deprotection occurs before protection
            has_exchange = any(deprotect_idx < protect_idx 
                             for deprotect_idx in tbdms_deprotection 
                             for protect_idx in benzyl_protection)
        
        return has_exchange, len(reactions)
    
    def detect_tbdms_deprotection(self, rxn):
        """Detect removal of TBDMS protecting group from phenol"""
        prod_smiles, react_smiles = rxn.split(">>")
        
        try:
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mol = Chem.MolFromSmiles(react_smiles.split(".")[0])  # Main reactant
            
            if not prod_mol or not react_mol:
                return False
            
            # TBDMS pattern: tert-butyldimethylsilyl group
            tbdms_pattern = Chem.MolFromSmarts("[Si](C)(C)C(C)(C)C")
            phenol_pattern = Chem.MolFromSmarts("c[OH]")
            tbdms_phenol_pattern = Chem.MolFromSmarts("c[O][Si](C)(C)C(C)(C)C")
            
            # Check if reactant has TBDMS-protected phenol and product has free phenol
            has_tbdms_reactant = react_mol.HasSubstructMatch(tbdms_phenol_pattern)
            has_free_phenol_product = prod_mol.HasSubstructMatch(phenol_pattern)
            has_tbdms_product = prod_mol.HasSubstructMatch(tbdms_pattern)
            
            return has_tbdms_reactant and has_free_phenol_product and not has_tbdms_product
            
        except:
            return False
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection of phenol"""
        prod_smiles, react_smiles = rxn.split(">>")
        
        try:
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mol = Chem.MolFromSmiles(react_smiles.split(".")[0])  # Main reactant
            
            if not prod_mol or not react_mol:
                return False
            
            # Benzyl ether pattern
            benzyl_ether_pattern = Chem.MolFromSmarts("c[O]Cc1ccccc1")
            phenol_pattern = Chem.MolFromSmarts("c[OH]")
            
            # Check if reactant has free phenol and product has benzyl ether
            has_free_phenol_reactant = react_mol.HasSubstructMatch(phenol_pattern)
            has_benzyl_ether_product = prod_mol.HasSubstructMatch(benzyl_ether_pattern)
            has_free_phenol_product = prod_mol.HasSubstructMatch(phenol_pattern)
            
            # Count phenol groups to ensure protection occurred
            reactant_phenols = len(react_mol.GetSubstructMatches(phenol_pattern))
            product_phenols = len(prod_mol.GetSubstructMatches(phenol_pattern))
            
            return (has_free_phenol_reactant and has_benzyl_ether_product and 
                   product_phenols < reactant_phenols)
            
        except:
            return False
