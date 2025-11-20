"""Generated evaluation code for: Sequential protecting group strategy with THP and TBS"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes for sequential protecting group strategy with THP and TBS.
    Checks for orthogonal use of THP and TBS protecting groups with sequential deprotection.
    """
    
    def __init__(self, config):
        self.required_groups = config["parameters"]["groups"]  # ["THP", "TBS"]
        self.approach = config["parameters"]["approach"]  # "orthogonal"
        self.deprotection_order = config["parameters"]["deprotection_order"]  # "sequential"
        
        # SMARTS patterns for protecting groups
        self.thp_pattern = Chem.MolFromSmarts("[CH2][CH2][CH2][CH2][CH2]O[CH]O")  # THP ether
        self.tbs_pattern = Chem.MolFromSmarts("[Si](C)(C)C(C)(C)C")  # TBS silyl ether
        
        # Deprotection patterns
        self.thp_deprotection = Chem.MolFromSmarts("[OH]")  # Reveals OH after THP removal
        self.tbs_deprotection = Chem.MolFromSmarts("[OH]")  # Reveals OH after TBS removal
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        thp_protection_step = -1
        tbs_protection_step = -1
        thp_deprotection_step = -1
        tbs_deprotection_step = -1
        
        for i, rxn in enumerate(reactions):
            # Check for protection reactions
            if self.detect_thp_protection(rxn):
                thp_protection_step = i
            if self.detect_tbs_protection(rxn):
                tbs_protection_step = i
                
            # Check for deprotection reactions
            if self.detect_thp_deprotection(rxn):
                thp_deprotection_step = i
            if self.detect_tbs_deprotection(rxn):
                tbs_deprotection_step = i
        
        # Evaluate strategy conditions
        has_both_groups = (thp_protection_step >= 0) and (tbs_protection_step >= 0)
        
        # Check orthogonal approach - both groups should be used
        orthogonal_condition = has_both_groups if self.approach == "orthogonal" else True
        
        # Check sequential deprotection - deprotections should occur in different steps
        sequential_condition = True
        if self.deprotection_order == "sequential" and thp_deprotection_step >= 0 and tbs_deprotection_step >= 0:
            sequential_condition = abs(thp_deprotection_step - tbs_deprotection_step) >= 1
        
        # Overall condition
        condition = orthogonal_condition and sequential_condition and has_both_groups
        
        return condition, len(reactions)
    
    def detect_thp_protection(self, rxn):
        """Detect THP protection reaction (formation of THP ether)"""
        reactants, products = self.parse_reaction_smiles(rxn)
        
        # Check if THP group appears in products but not reactants
        thp_in_products = any(mol.HasSubstructMatch(self.thp_pattern) for mol in products)
        thp_in_reactants = any(mol.HasSubstructMatch(self.thp_pattern) for mol in reactants)
        
        return thp_in_products and not thp_in_reactants
    
    def detect_tbs_protection(self, rxn):
        """Detect TBS protection reaction (formation of TBS ether)"""
        reactants, products = self.parse_reaction_smiles(rxn)
        
        # Check if TBS group appears in products but not reactants
        tbs_in_products = any(mol.HasSubstructMatch(self.tbs_pattern) for mol in products)
        tbs_in_reactants = any(mol.HasSubstructMatch(self.tbs_pattern) for mol in reactants)
        
        return tbs_in_products and not tbs_in_reactants
    
    def detect_thp_deprotection(self, rxn):
        """Detect THP deprotection reaction (removal of THP group)"""
        reactants, products = self.parse_reaction_smiles(rxn)
        
        # Check if THP group disappears from reactants to products
        thp_in_reactants = any(mol.HasSubstructMatch(self.thp_pattern) for mol in reactants)
        thp_in_products = any(mol.HasSubstructMatch(self.thp_pattern) for mol in products)
        
        return thp_in_reactants and not thp_in_products
    
    def detect_tbs_deprotection(self, rxn):
        """Detect TBS deprotection reaction (removal of TBS group)"""
        reactants, products = self.parse_reaction_smiles(rxn)
        
        # Check if TBS group disappears from reactants to products
        tbs_in_reactants = any(mol.HasSubstructMatch(self.tbs_pattern) for mol in reactants)
        tbs_in_products = any(mol.HasSubstructMatch(self.tbs_pattern) for mol in products)
        
        return tbs_in_reactants and not tbs_in_products
    
    def parse_reaction_smiles(self, rxn_smiles):
        """Parse reaction SMILES into reactant and product molecules"""
        parts = rxn_smiles.split(">>")
        reactant_smiles = parts[0].split(".")
        product_smiles = parts[1].split(".")
        
        reactants = [Chem.MolFromSmiles(smi) for smi in reactant_smiles if smi]
        products = [Chem.MolFromSmiles(smi) for smi in product_smiles if smi]
        
        return [mol for mol in reactants if mol is not None], [mol for mol in products if mol is not None]
