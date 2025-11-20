"""Generated evaluation code for: Sequential protecting group strategy for catechol formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a sequential protecting group strategy is used for catechol formation.
    Checks for the presence of MOM ether protection followed by methylenedioxy protection
    before final deprotection to form catechol.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.functional_group = config["parameters"]["functional_group"]
        self.strategy = config["parameters"]["strategy"]
        
        # Define SMARTS patterns for detecting protecting groups and functional groups
        self.mom_pattern = "[OH1][CH2]O[CH3]"  # MOM ether
        self.methylenedioxy_pattern = "O1[CH2]O*1"  # Methylenedioxy bridge
        self.catechol_pattern = "c1ccccc1[OH1][OH1]"  # Catechol (ortho-dihydroxybenzene)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track the sequence of protecting group operations
        mom_formation = False
        mom_to_methylenedioxy = False
        methylenedioxy_deprotection = False
        catechol_formed = False
        
        # Analyze reactions in chronological order (reverse of synthesis)
        for i, rxn in enumerate(reversed(reactions)):
            # Check for catechol formation (deprotection step)
            if self.detect_catechol_formation(rxn):
                catechol_formed = True
                
            # Check for methylenedioxy deprotection
            if self.detect_methylenedioxy_deprotection(rxn):
                methylenedioxy_deprotection = True
                
            # Check for MOM to methylenedioxy conversion
            if self.detect_mom_to_methylenedioxy(rxn):
                mom_to_methylenedioxy = True
                
            # Check for MOM ether formation
            if self.detect_mom_formation(rxn):
                mom_formation = True
        
        # Sequential strategy requires all steps in the correct sequence
        sequential_strategy = (mom_formation and 
                             mom_to_methylenedioxy and 
                             methylenedioxy_deprotection and 
                             catechol_formed)
        
        return sequential_strategy, len(reactions)
    
    def detect_mom_formation(self, rxn):
        """Detect MOM ether formation reaction"""
        reactants, products = self.parse_reaction(rxn)
        
        # Look for phenol in reactants and MOM ether in products
        phenol_in_reactants = any(self.has_phenol(mol) for mol in reactants)
        mom_in_products = any(self.has_substructure(mol, self.mom_pattern) for mol in products)
        
        return phenol_in_reactants and mom_in_products
    
    def detect_mom_to_methylenedioxy(self, rxn):
        """Detect conversion of MOM ether to methylenedioxy protection"""
        reactants, products = self.parse_reaction(rxn)
        
        mom_in_reactants = any(self.has_substructure(mol, self.mom_pattern) for mol in reactants)
        methylenedioxy_in_products = any(self.has_substructure(mol, self.methylenedioxy_pattern) for mol in products)
        
        return mom_in_reactants and methylenedioxy_in_products
    
    def detect_methylenedioxy_deprotection(self, rxn):
        """Detect methylenedioxy deprotection"""
        reactants, products = self.parse_reaction(rxn)
        
        methylenedioxy_in_reactants = any(self.has_substructure(mol, self.methylenedioxy_pattern) for mol in reactants)
        # Check if deprotection leads to diol or catechol precursor
        diol_in_products = any(self.has_diol(mol) for mol in products)
        
        return methylenedioxy_in_reactants and diol_in_products
    
    def detect_catechol_formation(self, rxn):
        """Detect final catechol formation"""
        reactants, products = self.parse_reaction(rxn)
        
        catechol_in_products = any(self.has_substructure(mol, self.catechol_pattern) for mol in products)
        return catechol_in_products
    
    def parse_reaction(self, rxn):
        """Parse reaction SMILES into reactant and product molecules"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(smi.strip()) for smi in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(smi.strip()) for smi in rxn_parts[1].split(".")]
        return [mol for mol in reactants if mol is not None], [mol for mol in products if mol is not None]
    
    def has_substructure(self, mol, pattern):
        """Check if molecule contains the specified substructure pattern"""
        if mol is None:
            return False
        pattern_mol = Chem.MolFromSmarts(pattern)
        if pattern_mol is None:
            return False
        return mol.HasSubstructMatch(pattern_mol)
    
    def has_phenol(self, mol):
        """Check if molecule contains phenol group"""
        if mol is None:
            return False
        phenol_pattern = Chem.MolFromSmarts("c[OH1]")
        return mol.HasSubstructMatch(phenol_pattern)
    
    def has_diol(self, mol):
        """Check if molecule contains diol functionality"""
        if mol is None:
            return False
        diol_pattern = Chem.MolFromSmarts("[OH1][C,c][C,c][OH1]")
        return mol.HasSubstructMatch(diol_pattern)
