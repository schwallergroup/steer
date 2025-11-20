"""Generated evaluation code for: Sequential protecting group cycling approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes that use sequential protecting group cycling approach.
    Checks for installation and removal of N-benzyl and N-Boc protecting groups
    in a specific sequence on the same nitrogen atom.
    """
    
    def __init__(self, config):
        self.sequential_protection = config.get("sequential_protection", True)
        self.n_benzyl_steps = config.get("n_benzyl_steps", [])
        self.boc_steps = config.get("boc_steps", [])
        
        # SMARTS patterns for protecting group detection
        self.benzyl_pattern = Chem.MolFromSmarts("[NH1,NH2]-[CH2]-c1ccccc1")  # N-benzyl
        self.boc_pattern = Chem.MolFromSmarts("[NH1,NH2]-C(=O)-O-C(C)(C)C")   # N-Boc
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        benzyl_installations = []
        benzyl_removals = []
        boc_installations = []
        boc_removals = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_benzyl_installation(rxn):
                benzyl_installations.append(i)
            elif self.detect_benzyl_removal(rxn):
                benzyl_removals.append(i)
            elif self.detect_boc_installation(rxn):
                boc_installations.append(i)
            elif self.detect_boc_removal(rxn):
                boc_removals.append(i)
        
        # Check if sequential protection pattern is followed
        condition_met = self.check_sequential_pattern(
            benzyl_installations, benzyl_removals, 
            boc_installations, boc_removals
        )
        
        return condition_met, len(reactions)
    
    def detect_benzyl_installation(self, rxn):
        """Detect N-benzyl protection installation"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        # Count benzyl groups in reactants vs products
        reactant_benzyl = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) 
                            for mol in reactants if mol)
        product_benzyl = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) 
                           for mol in products if mol)
        
        return product_benzyl > reactant_benzyl
    
    def detect_benzyl_removal(self, rxn):
        """Detect N-benzyl deprotection"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        # Count benzyl groups in reactants vs products
        reactant_benzyl = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) 
                            for mol in reactants if mol)
        product_benzyl = sum(len(mol.GetSubstructMatches(self.benzyl_pattern)) 
                           for mol in products if mol)
        
        return reactant_benzyl > product_benzyl
    
    def detect_boc_installation(self, rxn):
        """Detect N-Boc protection installation"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        # Count Boc groups in reactants vs products
        reactant_boc = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                         for mol in reactants if mol)
        product_boc = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                        for mol in products if mol)
        
        return product_boc > reactant_boc
    
    def detect_boc_removal(self, rxn):
        """Detect N-Boc deprotection"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        # Count Boc groups in reactants vs products
        reactant_boc = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                         for mol in reactants if mol)
        product_boc = sum(len(mol.GetSubstructMatches(self.boc_pattern)) 
                        for mol in products if mol)
        
        return reactant_boc > product_boc
    
    def check_sequential_pattern(self, benzyl_inst, benzyl_rem, boc_inst, boc_rem):
        """
        Check if the protecting group operations follow the expected sequential pattern:
        N-benzyl installation -> N-benzyl removal -> N-Boc installation
        """
        if not self.sequential_protection:
            # Just check that the required steps are present
            return (len(benzyl_inst) > 0 and len(benzyl_rem) > 0 and len(boc_inst) > 0)
        
        # For sequential protection, check timing
        if not (benzyl_inst and benzyl_rem and boc_inst):
            return False
        
        # Find the sequence: benzyl installation, then removal, then boc installation
        for b_inst in benzyl_inst:
            for b_rem in benzyl_rem:
                for boc_i in boc_inst:
                    # Check if sequence is correct (later steps have higher indices)
                    if b_inst < b_rem < boc_i:
                        return True
        
        return False
