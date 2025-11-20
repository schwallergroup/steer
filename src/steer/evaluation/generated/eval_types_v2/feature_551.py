"""Generated evaluation code for: Non-selective silyl ether deprotection sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NonSelectiveSilylEtherDeprotection(MultiRxnCondBase):
    """
    Detects non-selective silyl ether deprotection sequences where TBDPS is removed
    before TBS, which typically lacks reliable selectivity.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.selectivity = config["parameters"]["selectivity"]
        self.deprotection_order = config["parameters"]["deprotection_order"]
        
        # SMARTS patterns for silyl ethers
        self.tbdps_pattern = "[O:1][Si]([CH2][c:2]1[cH][cH][cH][cH][cH]1)([CH2][c:3]1[cH][cH][cH][cH][cH]1)[C]([CH3])([CH3])[CH3]"  # TBDPS
        self.tbs_pattern = "[O:1][Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]"  # TBS
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if route contains problematic TBDPS-before-TBS deprotection sequence"""
        reactions = self.get_rxns(d)
        
        tbdps_deprotection_step = -1
        tbs_deprotection_step = -1
        
        # Check each reaction for deprotection events
        for i, rxn in enumerate(reactions):
            if self.detect_tbdps_deprotection(rxn):
                tbdps_deprotection_step = i
            if self.detect_tbs_deprotection(rxn):
                tbs_deprotection_step = i
        
        # Check if both protecting groups are present and TBDPS is removed before TBS
        if tbdps_deprotection_step >= 0 and tbs_deprotection_step >= 0:
            if tbdps_deprotection_step < tbs_deprotection_step:
                # Found problematic sequence - return early step for higher penalty
                condition = True
                return condition, min(tbdps_deprotection_step, tbs_deprotection_step)
        
        return False, len(reactions)
    
    def detect_tbdps_deprotection(self, rxn):
        """Detect TBDPS deprotection (TBDPS present in reactants but not products)"""
        reactants, products = rxn.split(">>")
        
        # Check if TBDPS is in reactants
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        has_tbdps_reactant = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbdps_pattern)) 
                                for mol in reactant_mols if mol)
        
        # Check if TBDPS is absent in products (or reduced count)
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
        tbdps_count_products = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.tbdps_pattern))) 
                                  for mol in product_mols if mol)
        tbdps_count_reactants = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.tbdps_pattern))) 
                                   for mol in reactant_mols if mol)
        
        return has_tbdps_reactant and tbdps_count_products < tbdps_count_reactants
    
    def detect_tbs_deprotection(self, rxn):
        """Detect TBS deprotection (TBS present in reactants but not products)"""
        reactants, products = rxn.split(">>")
        
        # Check if TBS is in reactants
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        has_tbs_reactant = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbs_pattern)) 
                              for mol in reactant_mols if mol)
        
        # Check if TBS is absent in products (or reduced count)
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
        tbs_count_products = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.tbs_pattern))) 
                                for mol in product_mols if mol)
        tbs_count_reactants = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.tbs_pattern))) 
                                 for mol in reactant_mols if mol)
        
        return has_tbs_reactant and tbs_count_products < tbs_count_reactants
