"""Generated evaluation code for: Benzyl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates benzyl protecting group strategy for phenol groups.
    Checks if phenol is protected with benzyl group and remains protected 
    for the specified number of steps before deprotection.
    """
    
    def __init__(self, config):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.target_steps_protected = config["parameters"]["steps_protected"]
        
        # SMARTS patterns
        self.phenol_pattern = "[OH1][c]"  # Phenol OH
        self.benzyl_ether_pattern = "[CH2][c]1[cH][cH][cH][cH][cH]1"  # Benzyl group
        self.benzyl_phenol_ether_pattern = "[OH0]([CH2][c]1[cH][cH][cH][cH][cH]1)[c]"  # Benzyl-protected phenol
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find protection step
        protection_step = -1
        deprotection_step = -1
        
        for i, rxn in enumerate(reactions):
            if self.is_protection_step(rxn):
                protection_step = i
            elif self.is_deprotection_step(rxn) and protection_step >= 0:
                deprotection_step = i
                break
        
        # Check if strategy is followed correctly
        if protection_step >= 0:
            if deprotection_step >= 0:
                steps_protected = deprotection_step - protection_step
                condition = steps_protected >= self.target_steps_protected
            else:
                # Protection found but no deprotection yet
                steps_protected = len(reactions) - protection_step
                condition = steps_protected >= self.target_steps_protected
        else:
            condition = False
        
        return condition, len(reactions)
    
    def is_protection_step(self, rxn):
        """Check if reaction involves benzyl protection of phenol"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        if not prod_mol or not all(react_mols):
            return False
        
        # Check if product has benzyl-protected phenol
        has_protected_phenol = prod_mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_phenol_ether_pattern))
        
        # Check if any reactant has free phenol
        has_free_phenol = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) for mol in react_mols)
        
        # Check if benzyl reagent is present in reactants
        has_benzyl_reagent = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)) for mol in react_mols)
        
        return has_protected_phenol and has_free_phenol and has_benzyl_reagent
    
    def is_deprotection_step(self, rxn):
        """Check if reaction involves benzyl deprotection to reveal phenol"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        if not prod_mol or not all(react_mols):
            return False
        
        # Check if product has free phenol
        has_free_phenol = prod_mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern))
        
        # Check if any reactant has benzyl-protected phenol
        has_protected_phenol = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_phenol_ether_pattern)) for mol in react_mols)
        
        return has_free_phenol and has_protected_phenol
    
    def route_scoring(self, x):
        """Score based on whether the protecting group strategy was used correctly"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 1 - x  # Earlier implementation is better
