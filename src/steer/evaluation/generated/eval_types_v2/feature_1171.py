"""Generated evaluation code for: Four-step protecting group cycling sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes that use a four-step protecting group cycling sequence.
    Detects protect-esterify-deprotect-esterify patterns with specified protecting groups.
    """
    
    def __init__(self, config):
        self.protection_steps = config.get("protection_steps", 4)
        self.sequence_type = config.get("sequence_type", "cycling")
        self.protecting_groups = config.get("protecting_groups", ["tert-butyl ester"])
        
        # SMARTS patterns for protecting groups
        self.pg_patterns = {
            "tert-butyl ester": "[CX3](=O)[OX2]C(C)(C)C",
            "benzyl ester": "[CX3](=O)[OX2]Cc1ccccc1",
            "methyl ester": "[CX3](=O)[OX2]C",
            "ethyl ester": "[CX3](=O)[OX2]CC"
        }
        
        # Carboxylic acid pattern for deprotection detection
        self.carboxylic_acid_pattern = "[CX3](=O)[OX2H1]"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Need at least 4 reactions for the cycling sequence
        if len(reactions) < self.protection_steps:
            return False, len(reactions)
        
        # Look for the protecting group cycling pattern
        cycling_found = self.detect_cycling_sequence(reactions)
        
        return cycling_found, len(reactions)
    
    def detect_cycling_sequence(self, reactions) -> bool:
        """
        Detect the four-step cycling sequence: protect-esterify-deprotect-esterify
        """
        for i in range(len(reactions) - 3):
            # Check if we have a valid 4-step sequence starting at position i
            if self.is_valid_cycling_sequence(reactions[i:i+4]):
                return True
        return False
    
    def is_valid_cycling_sequence(self, four_reactions) -> bool:
        """
        Check if four consecutive reactions form a protecting group cycling sequence
        """
        if len(four_reactions) != 4:
            return False
            
        step1, step2, step3, step4 = four_reactions
        
        # Step 1: Protection (should introduce protecting group)
        protection_step = self.is_protection_step(step1)
        
        # Step 2: First esterification
        first_esterify = self.is_esterification_step(step2)
        
        # Step 3: Deprotection (should remove protecting group, reveal carboxylic acid)
        deprotection_step = self.is_deprotection_step(step3)
        
        # Step 4: Second esterification
        second_esterify = self.is_esterification_step(step4)
        
        return protection_step and first_esterify and deprotection_step and second_esterify
    
    def is_protection_step(self, rxn) -> bool:
        """Check if reaction introduces a protecting group"""
        reactants_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        product_mol = Chem.MolFromSmiles(rxn[0])
        
        # Count protecting groups in reactants vs product
        reactant_pg_count = sum(self.count_protecting_groups(mol) for mol in reactants_mols)
        product_pg_count = self.count_protecting_groups(product_mol)
        
        return product_pg_count > reactant_pg_count
    
    def is_deprotection_step(self, rxn) -> bool:
        """Check if reaction removes protecting group and reveals carboxylic acid"""
        reactants_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        product_mol = Chem.MolFromSmiles(rxn[0])
        
        # Count protecting groups in reactants vs product (should decrease)
        reactant_pg_count = sum(self.count_protecting_groups(mol) for mol in reactants_mols)
        product_pg_count = self.count_protecting_groups(product_mol)
        
        # Count carboxylic acids (should increase)
        reactant_acid_count = sum(self.count_carboxylic_acids(mol) for mol in reactants_mols)
        product_acid_count = self.count_carboxylic_acids(product_mol)
        
        return (product_pg_count < reactant_pg_count) and (product_acid_count > reactant_acid_count)
    
    def is_esterification_step(self, rxn) -> bool:
        """Check if reaction forms an ester bond"""
        reactants_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        product_mol = Chem.MolFromSmiles(rxn[0])
        
        # Look for carboxylic acid in reactants
        has_acid_reactant = any(self.count_carboxylic_acids(mol) > 0 for mol in reactants_mols)
        
        # Look for ester formation (simple ester pattern)
        simple_ester_pattern = "[CX3](=O)[OX2][CX4]"
        simple_ester = Chem.MolFromSmarts(simple_ester_pattern)
        
        reactant_ester_count = sum(len(mol.GetSubstructMatches(simple_ester)) if mol else 0 for mol in reactants_mols)
        product_ester_count = len(product_mol.GetSubstructMatches(simple_ester)) if product_mol else 0
        
        return has_acid_reactant and (product_ester_count > reactant_ester_count)
    
    def count_protecting_groups(self, mol) -> int:
        """Count the number of specified protecting groups in the molecule"""
        if not mol:
            return 0
            
        total_count = 0
        for pg_name in self.protecting_groups:
            if pg_name in self.pg_patterns:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                if pattern:
                    total_count += len(mol.GetSubstructMatches(pattern))
        
        return total_count
    
    def count_carboxylic_acids(self, mol) -> int:
        """Count the number of carboxylic acid groups in the molecule"""
        if not mol:
            return 0
            
        acid_pattern = Chem.MolFromSmarts(self.carboxylic_acid_pattern)
        if acid_pattern:
            return len(mol.GetSubstructMatches(acid_pattern))
        return 0
