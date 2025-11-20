"""Generated evaluation code for: Dual protecting group strategy for cephalosporin"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates dual protecting group strategy for cephalosporin synthesis.
    Checks for simultaneous use of benzhydryl ester (carboxylic acid protection)
    and Boc (amine protection) groups with coordinated deprotection.
    """
    
    def __init__(self, config):
        self.functional_groups = config.get("functional_groups", ["carboxylic_acid", "amine"])
        self.protecting_groups = config.get("protecting_groups", ["benzhydryl", "boc"])
        self.simultaneous_deprotection = config.get("simultaneous_deprotection", True)
        
        # SMARTS patterns for functional groups
        self.fg_patterns = {
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "amine": "[NX3;H2,H1;!$(NC=O)]"
        }
        
        # SMARTS patterns for protecting groups
        self.pg_patterns = {
            "benzhydryl": "[CH1](c1ccccc1)(c2ccccc2)[OX2][CX3]=O",  # Benzhydryl ester
            "boc": "[CX3](=O)[OX2][CX4]([CH3])([CH3])[CH3].[NX3]"    # Boc-protected amine
        }
        
        # Cephalosporin core pattern
        self.cephalosporin_core = "[C@@H]1[C@H]([NH1][CX3]=O)C2=C(CS1)[CH2][CH2]S[C@@H]2[CX3](=O)[OH1]"

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection and deprotection events
        benzhydryl_protection = False
        boc_protection = False
        dual_deprotection = False
        
        # Check each reaction for protection/deprotection events
        for rxn in reactions:
            if self.detect_benzhydryl_protection(rxn):
                benzhydryl_protection = True
            if self.detect_boc_protection(rxn):
                boc_protection = True
            if self.detect_dual_deprotection(rxn):
                dual_deprotection = True
        
        # Strategy is successful if both protecting groups are used
        # and simultaneous deprotection occurs (if required)
        condition = (benzhydryl_protection and boc_protection and 
                    (not self.simultaneous_deprotection or dual_deprotection))
        
        return condition, len(reactions)
    
    def detect_benzhydryl_protection(self, rxn):
        """Detect formation of benzhydryl ester from carboxylic acid"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        # Check if product has benzhydryl ester
        benzhydryl_pattern = Chem.MolFromSmarts(self.pg_patterns["benzhydryl"])
        if not prod_mol.HasSubstructMatch(benzhydryl_pattern):
            return False
        
        # Check if reactants had free carboxylic acid
        carboxyl_pattern = Chem.MolFromSmarts(self.fg_patterns["carboxylic_acid"])
        return any(mol.HasSubstructMatch(carboxyl_pattern) for mol in react_mols)
    
    def detect_boc_protection(self, rxn):
        """Detect formation of Boc-protected amine"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        # Check if product has Boc group
        boc_pattern = Chem.MolFromSmarts("[NX3][CX3](=O)[OX2][CX4]([CH3])([CH3])[CH3]")
        if not prod_mol.HasSubstructMatch(boc_pattern):
            return False
        
        # Check if reactants had free amine
        amine_pattern = Chem.MolFromSmarts(self.fg_patterns["amine"])
        return any(mol.HasSubstructMatch(amine_pattern) for mol in react_mols)
    
    def detect_dual_deprotection(self, rxn):
        """Detect simultaneous removal of both protecting groups"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        # Check if reactants have both protecting groups
        has_benzhydryl = False
        has_boc = False
        
        benzhydryl_pattern = Chem.MolFromSmarts(self.pg_patterns["benzhydryl"])
        boc_pattern = Chem.MolFromSmarts("[NX3][CX3](=O)[OX2][CX4]([CH3])([CH3])[CH3]")
        
        for mol in react_mols:
            if mol.HasSubstructMatch(benzhydryl_pattern):
                has_benzhydryl = True
            if mol.HasSubstructMatch(boc_pattern):
                has_boc = True
        
        if not (has_benzhydryl and has_boc):
            return False
        
        # Check if product has deprotected functional groups
        carboxyl_pattern = Chem.MolFromSmarts(self.fg_patterns["carboxylic_acid"])
        amine_pattern = Chem.MolFromSmarts(self.fg_patterns["amine"])
        
        return (prod_mol.HasSubstructMatch(carboxyl_pattern) and 
                prod_mol.HasSubstructMatch(amine_pattern))
    
    def route_scoring(self, x):
        """Score based on successful implementation of dual protecting group strategy"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 10 - x * 2  # Earlier implementation is better, max score 10
