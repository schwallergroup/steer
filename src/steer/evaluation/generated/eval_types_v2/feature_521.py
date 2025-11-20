"""Generated evaluation code for: Extensive protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes based on protecting group cycling strategy.
    Detects cycles of protecting/deprotecting reactions for specified functional groups.
    """
    
    def __init__(self, config):
        self.protection_cycles = config.get("protection_cycles", 2)
        self.functional_groups = config.get("functional_groups", ["carboxylic_acid", "ester"])
        self.cycle_length = config.get("cycle_length", 6)
        
        # Define SMARTS patterns for functional groups and their protecting group variants
        self.fg_patterns = {
            "carboxylic_acid": "[CX3](=O)[OH]",
            "ester": "[CX3](=O)[OX2][CX4]",
            "acyl_chloride": "[CX3](=O)[Cl]",
            "methyl_ester": "[CX3](=O)[OX2][CH3]",
            "t_butyl_ester": "[CX3](=O)[OX2]C(C)(C)C"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group transformations in sequence
        pg_sequence = self.detect_protecting_group_sequence(reactions)
        
        # Count complete cycles
        cycles_found = self.count_protection_cycles(pg_sequence)
        
        # Check if we meet the cycle requirements
        condition = (cycles_found >= self.protection_cycles and 
                    len(pg_sequence) >= self.cycle_length)
        
        return condition, len(reactions)
    
    def detect_protecting_group_sequence(self, reactions):
        """Detect sequence of protecting group transformations"""
        sequence = []
        
        for rxn in reactions:
            transformation = self.classify_pg_transformation(rxn)
            if transformation:
                sequence.append(transformation)
        
        return sequence
    
    def classify_pg_transformation(self, rxn_smiles):
        """Classify a reaction as a specific protecting group transformation"""
        try:
            rxn_parts = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
            
            if not all(reactants + products):
                return None
            
            # Check for specific transformations
            reactant_fgs = self.get_functional_groups(reactants)
            product_fgs = self.get_functional_groups(products)
            
            # Detect transformation patterns
            if "carboxylic_acid" in reactant_fgs and "methyl_ester" in product_fgs:
                return "acid_to_methyl_ester"
            elif "methyl_ester" in reactant_fgs and "carboxylic_acid" in product_fgs:
                return "methyl_ester_to_acid"
            elif "carboxylic_acid" in reactant_fgs and "acyl_chloride" in product_fgs:
                return "acid_to_acyl_chloride"
            elif "acyl_chloride" in reactant_fgs and "t_butyl_ester" in product_fgs:
                return "acyl_chloride_to_tbutyl_ester"
            elif "t_butyl_ester" in reactant_fgs and "carboxylic_acid" in product_fgs:
                return "tbutyl_ester_to_acid"
            elif "ester" in reactant_fgs and "carboxylic_acid" in product_fgs:
                return "ester_hydrolysis"
            elif "carboxylic_acid" in reactant_fgs and "ester" in product_fgs:
                return "esterification"
                
        except Exception:
            pass
        
        return None
    
    def get_functional_groups(self, mols):
        """Identify functional groups present in molecules"""
        found_fgs = set()
        
        for mol in mols:
            if mol is None:
                continue
                
            for fg_name, pattern in self.fg_patterns.items():
                if fg_name in self.functional_groups or fg_name in ["acyl_chloride", "methyl_ester", "t_butyl_ester"]:
                    smarts_mol = Chem.MolFromSmarts(pattern)
                    if smarts_mol and mol.HasSubstructMatch(smarts_mol):
                        found_fgs.add(fg_name)
        
        return found_fgs
    
    def count_protection_cycles(self, sequence):
        """Count complete protecting group cycles in the sequence"""
        if len(sequence) < 3:
            return 0
        
        cycles = 0
        cycle_patterns = [
            ["acid_to_methyl_ester", "methyl_ester_to_acid", "acid_to_acyl_chloride", 
             "acyl_chloride_to_tbutyl_ester", "tbutyl_ester_to_acid", "acid_to_methyl_ester"],
            ["esterification", "ester_hydrolysis", "esterification"],
            ["acid_to_acyl_chloride", "acyl_chloride_to_tbutyl_ester", "tbutyl_ester_to_acid"]
        ]
        
        # Look for repeating patterns
        for i in range(len(sequence) - 2):
            for pattern in cycle_patterns:
                if self.matches_cycle_pattern(sequence[i:], pattern):
                    cycles += 1
                    break
        
        return cycles
    
    def matches_cycle_pattern(self, sequence, pattern):
        """Check if sequence matches a specific cycle pattern"""
        if len(sequence) < len(pattern):
            return False
        
        for i, expected in enumerate(pattern):
            if i >= len(sequence) or sequence[i] != expected:
                return False
        
        return True
