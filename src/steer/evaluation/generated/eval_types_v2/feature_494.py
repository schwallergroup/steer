"""Generated evaluation code for: Four-step protecting group manipulation sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSequence(MultiRxnCondBase):
    """
    Evaluates routes for a specific protecting group manipulation sequence.
    Checks for a four-step protect-deprotect-protect-deprotect pattern
    involving benzoyl, TBDMS, and benzyl protecting groups.
    """
    
    def __init__(self, config):
        self.sequence_length = config.get("sequence_length", 4)
        self.target_operations = config.get("operations", ["protect", "deprotect", "protect", "deprotect"])
        self.allowed_groups = config.get("groups", ["benzoyl", "TBDMS", "benzyl"])
        
        # Define protecting group SMARTS patterns
        self.pg_patterns = {
            "benzoyl": "[OH1,NH1,NH2]-C(=O)-c1ccccc1",  # Benzoyl ester/amide
            "TBDMS": "[OH1,NH1]-[Si](C)(C)C(C)(C)C",     # TBDMS silyl ether/amine
            "benzyl": "[OH1,NH1]-Cc1ccccc1"              # Benzyl ether/amine
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        pg_operations = []
        
        # Analyze each reaction for protecting group operations
        for rxn in reactions:
            operation = self.classify_pg_operation(rxn)
            if operation:
                pg_operations.append(operation)
        
        # Check if we found the target sequence
        condition_met = self.matches_target_sequence(pg_operations)
        return condition_met, len(reactions)
    
    def classify_pg_operation(self, rxn):
        """
        Classify a reaction as protect/deprotect for allowed protecting groups.
        Returns tuple (operation_type, group) or None if not a PG operation.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not reactants or not products:
            return None
        
        for group_name, pattern in self.pg_patterns.items():
            if group_name not in self.allowed_groups:
                continue
                
            pattern_mol = Chem.MolFromSmarts(pattern)
            if not pattern_mol:
                continue
            
            reactant_matches = reactants.GetSubstructMatches(pattern_mol)
            product_matches = products.GetSubstructMatches(pattern_mol)
            
            # Protection: PG appears in products but not reactants (or more in products)
            if len(product_matches) > len(reactant_matches):
                return ("protect", group_name)
            
            # Deprotection: PG appears in reactants but not products (or fewer in products)
            elif len(reactant_matches) > len(product_matches):
                return ("deprotect", group_name)
        
        return None
    
    def matches_target_sequence(self, pg_operations):
        """
        Check if the found protecting group operations match the target sequence.
        """
        if len(pg_operations) != self.sequence_length:
            return False
        
        # Extract just the operation types (protect/deprotect)
        operation_types = [op[0] for op in pg_operations]
        
        # Check if the sequence matches the target pattern
        if operation_types == self.target_operations:
            # Verify that different protecting groups are used
            groups_used = [op[1] for op in pg_operations]
            unique_groups = set(groups_used)
            
            # Require at least 2 different protecting groups for meaningful sequence
            return len(unique_groups) >= 2
        
        return False
    
    def route_scoring(self, x):
        """
        Score based on whether the protecting group sequence was found.
        """
        if x < 0:
            return 0  # Sequence not found
        else:
            return 10  # Sequence found - full points
