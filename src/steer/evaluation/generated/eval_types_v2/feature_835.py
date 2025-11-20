"""Generated evaluation code for: Sequential dual protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialDualProtectingGroup(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses a sequential dual protecting group strategy
    with TBDMS and trityl groups to differentiate secondary and primary alcohols.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "sequential_dual")
        self.protecting_groups = config.get("protecting_groups", ["TBDMS", "trityl"])
        self.functional_groups = config.get("functional_groups", ["secondary_alcohol", "primary_alcohol"])
        
        # SMARTS patterns for protecting groups
        self.tbdms_pattern = "[Si](C)(C)(C)C(C)(C)C"  # TBDMS group
        self.trityl_pattern = "C(c1ccccc1)(c2ccccc2)c3ccccc3"  # Trityl group
        
        # SMARTS patterns for alcohols
        self.primary_alcohol_pattern = "[CH2][OH]"  # Primary alcohol
        self.secondary_alcohol_pattern = "[CH]([OH])"  # Secondary alcohol
        
        # Protected alcohol patterns
        self.tbdms_protected_pattern = "[OH0][Si](C)(C)(C)C(C)(C)C"
        self.trityl_protected_pattern = "[OH0]C(c1ccccc1)(c2ccccc2)c3ccccc3"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group installation and removal
        tbdms_install = False
        trityl_install = False
        tbdms_remove = False
        trityl_remove = False
        sequential_order = False
        
        # Check each reaction for protecting group operations
        for i, rxn in enumerate(reactions):
            if self.detect_protection_reaction(rxn, "TBDMS"):
                tbdms_install = True
            elif self.detect_protection_reaction(rxn, "trityl"):
                trityl_install = True
            elif self.detect_deprotection_reaction(rxn, "TBDMS"):
                tbdms_remove = True
            elif self.detect_deprotection_reaction(rxn, "trityl"):
                trityl_remove = True
        
        # Check for sequential installation (both groups used)
        if tbdms_install and trityl_install:
            sequential_order = self.check_sequential_order(reactions)
        
        # Strategy is successful if both groups are installed sequentially
        # and at least one is selectively removed
        condition = (tbdms_install and trityl_install and sequential_order and 
                    (tbdms_remove or trityl_remove))
        
        return condition, len(reactions)
    
    def detect_protection_reaction(self, rxn, protecting_group):
        """Detect installation of a specific protecting group"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        if protecting_group == "TBDMS":
            pattern = Chem.MolFromSmarts(self.tbdms_protected_pattern)
            alcohol_pattern = Chem.MolFromSmarts("[OH]")
        else:  # trityl
            pattern = Chem.MolFromSmarts(self.trityl_protected_pattern)
            alcohol_pattern = Chem.MolFromSmarts("[OH]")
        
        # Check if product has protected group and reactant has free alcohol
        has_protected = prod_mol.HasSubstructMatch(pattern) if prod_mol else False
        has_free_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) for mol in react_mols if mol)
        
        return has_protected and has_free_alcohol
    
    def detect_deprotection_reaction(self, rxn, protecting_group):
        """Detect removal of a specific protecting group"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        if protecting_group == "TBDMS":
            pattern = Chem.MolFromSmarts(self.tbdms_protected_pattern)
        else:  # trityl
            pattern = Chem.MolFromSmarts(self.trityl_protected_pattern)
        
        alcohol_pattern = Chem.MolFromSmarts("[OH]")
        
        # Check if reactant has protected group and product has free alcohol
        has_protected = any(mol.HasSubstructMatch(pattern) for mol in react_mols if mol)
        has_free_alcohol = prod_mol.HasSubstructMatch(alcohol_pattern) if prod_mol else False
        
        return has_protected and has_free_alcohol
    
    def check_sequential_order(self, reactions):
        """Check if protecting groups are installed in a meaningful sequential manner"""
        tbdms_reactions = []
        trityl_reactions = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_protection_reaction(rxn, "TBDMS"):
                tbdms_reactions.append(i)
            elif self.detect_protection_reaction(rxn, "trityl"):
                trityl_reactions.append(i)
        
        # Sequential means they don't happen simultaneously and there's differentiation
        return len(tbdms_reactions) > 0 and len(trityl_reactions) > 0
    
    def route_scoring(self, x):
        """Score based on successful implementation of dual protecting group strategy"""
        if x < 0:
            return 0  # Strategy not found
        else:
            return 1 - x  # Earlier implementation is better
