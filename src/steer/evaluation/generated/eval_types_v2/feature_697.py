"""Generated evaluation code for: Protecting group cycling on oxygen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for protecting group cycling on oxygen atoms.
    Checks if the same oxygen undergoes protection, deprotection, and re-protection
    with different protecting groups in the specified sequence.
    """
    
    def __init__(self, config):
        self.atom_type = config["parameters"]["atom_type"]
        self.pattern = config["parameters"]["pattern"]
        self.groups = config["parameters"]["groups"]
        self.step_count = config["parameters"]["step_count"]
        
        # Define SMARTS patterns for protecting groups
        self.group_patterns = {
            "Bn": "[OH1][CH2]c1ccccc1",  # Benzyl ether
            "H": "[OH1]",  # Free hydroxyl
            "TBDMS": "[OH1][Si](C)(C)C(C)(C)C",  # TBDMS ether
            "Ac": "[OH1]C(=O)C",  # Acetyl ester
            "TBS": "[OH1][Si](C)(C)C(C)(C)C",  # TBS ether (similar to TBDMS)
            "PMB": "[OH1][CH2]c1ccc(OC)cc1",  # PMB ether
            "THP": "[OH1]C1OCCCC1"  # THP ether
        }
    
    def condition_depth(self, d):
        """Check if protecting group cycling pattern is found in the route."""
        reactions = self.get_rxns(d)
        
        if len(reactions) < self.step_count:
            return False, len(reactions)
        
        # Track oxygen atoms and their protecting group states through the route
        cycling_found = self.detect_protecting_group_cycling(reactions)
        
        return cycling_found, len(reactions)
    
    def detect_protecting_group_cycling(self, reactions):
        """Detect if the specified protecting group cycling pattern occurs."""
        # Track oxygen atom map numbers and their protecting group states
        oxygen_states = {}  # {atom_map_num: [list of protecting group states]}
        
        for rxn_smiles in reactions:
            self.update_oxygen_states(rxn_smiles, oxygen_states)
        
        # Check if any oxygen atom follows the specified cycling pattern
        for atom_map, states in oxygen_states.items():
            if len(states) >= self.step_count:
                if self.matches_cycling_pattern(states):
                    return True
        
        return False
    
    def update_oxygen_states(self, rxn_smiles, oxygen_states):
        """Update the protecting group states of oxygen atoms based on reaction."""
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            
            for product in products:
                if product is None:
                    continue
                    
                # Find oxygen atoms and their current protecting group state
                for atom in product.GetAtoms():
                    if atom.GetSymbol() == "O" and atom.GetAtomMapNum() > 0:
                        map_num = atom.GetAtomMapNum()
                        protecting_group = self.identify_protecting_group(product, atom)
                        
                        if map_num not in oxygen_states:
                            oxygen_states[map_num] = []
                        
                        # Only add if it's a different state than the last one
                        if not oxygen_states[map_num] or oxygen_states[map_num][-1] != protecting_group:
                            oxygen_states[map_num].append(protecting_group)
                            
        except Exception:
            pass  # Skip problematic reactions
    
    def identify_protecting_group(self, mol, oxygen_atom):
        """Identify which protecting group is on the oxygen atom."""
        atom_idx = oxygen_atom.GetIdx()
        
        # Create a molecule fragment around the oxygen for pattern matching
        for group_name, pattern_smarts in self.group_patterns.items():
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                # Check if this match includes our specific oxygen
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    if atom_idx in match:
                        return group_name
        
        # Default to unknown if no pattern matches
        return "unknown"
    
    def matches_cycling_pattern(self, states):
        """Check if the sequence of states matches the expected cycling pattern."""
        if len(states) < len(self.groups):
            return False
        
        # Look for the sequence of protecting groups in order
        for i in range(len(states) - len(self.groups) + 1):
            sequence = states[i:i + len(self.groups)]
            if sequence == self.groups:
                return True
        
        return False
    
    def route_scoring(self, x):
        """Convert condition result to scoring value."""
        if x < 0:
            return 0  # Pattern not found
        else:
            return 1 - x  # Earlier occurrence is better
