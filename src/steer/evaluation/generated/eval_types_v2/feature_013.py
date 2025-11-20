"""Generated evaluation code for: Triple orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TripleOrthogonalProtectingGroup(MultiRxnCondBase):
    """
    Evaluates synthesis routes for triple orthogonal protecting group strategy.
    Checks if TBS, Boc, and benzyl protecting groups are used simultaneously
    in the synthesis route to control regioselectivity.
    """
    
    def __init__(self, config):
        self.required_groups = config.get("protecting_groups", ["TBS", "Boc", "benzyl"])
        self.orthogonal = config.get("orthogonal", True)
        self.min_count = config.get("count", 3)
        
        # SMARTS patterns for protecting groups
        self.protection_patterns = {
            "TBS": "[Si](C)(C)C(C)(C)C",  # tert-butyldimethylsilyl
            "Boc": "CC(C)(C)OC(=O)",      # tert-butoxycarbonyl
            "benzyl": "Cc1ccccc1CO"       # benzyl group
        }
    
    def condition_depth(self, d):
        """Check if triple orthogonal protecting group strategy is employed"""
        reactions = self.get_rxns(d)
        
        # Track protecting groups found in the route
        groups_found = set()
        max_simultaneous = 0
        
        for rxn in reactions:
            current_groups = self.detect_protecting_groups_in_reaction(rxn)
            groups_found.update(current_groups)
            
            # Check for simultaneous presence in intermediates
            simultaneous_count = self.count_simultaneous_groups(rxn)
            max_simultaneous = max(max_simultaneous, simultaneous_count)
        
        # Condition met if we have all required groups and appropriate simultaneity
        required_groups_present = all(group in groups_found for group in self.required_groups)
        sufficient_simultaneity = max_simultaneous >= self.min_count
        
        condition = required_groups_present and sufficient_simultaneity
        return condition, len(reactions)
    
    def detect_protecting_groups_in_reaction(self, rxn):
        """Detect which protecting groups are involved in a reaction"""
        groups_found = set()
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return groups_found
            
        # Check both reactants and products
        all_molecules = rxn_parts[0].split(".") + rxn_parts[1].split(".")
        
        for mol_smiles in all_molecules:
            try:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol is None:
                    continue
                    
                for group_name, pattern in self.protection_patterns.items():
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        groups_found.add(group_name)
            except:
                continue
                
        return groups_found
    
    def count_simultaneous_groups(self, rxn):
        """Count maximum number of protecting groups present simultaneously in any molecule"""
        max_count = 0
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return max_count
            
        # Check all molecules in the reaction
        all_molecules = rxn_parts[0].split(".") + rxn_parts[1].split(".")
        
        for mol_smiles in all_molecules:
            try:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol is None:
                    continue
                    
                current_count = 0
                for group_name in self.required_groups:
                    pattern = self.protection_patterns.get(group_name)
                    if pattern:
                        pattern_mol = Chem.MolFromSmarts(pattern)
                        if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                            current_count += 1
                            
                max_count = max(max_count, current_count)
            except:
                continue
                
        return max_count
    
    def route_scoring(self, x):
        """Score based on successful implementation of orthogonal protecting group strategy"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            # Earlier implementation of orthogonal strategy is better
            return 1 - x
