"""Generated evaluation code for: Protecting group cycling on nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects protecting group cycling on nitrogen atoms where protection,
    deprotection, and reprotection occurs on the same nitrogen.
    """
    
    def __init__(self, config):
        self.atom_type = config["atom_type"]
        self.pattern = config["pattern"]
        self.groups = config["groups"]
        self.step_count = config["step_count"]
        
        # Define SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "PMB": "[N]-[CH2]-c1ccc(OC)cc1",  # p-methoxybenzyl
            "Bn": "[N]-[CH2]-c1ccccc1",       # benzyl
            "H": "[NH]",                       # free amine
            "Boc": "[N]-C(=O)-O-C(C)(C)C",   # tert-butoxycarbonyl
            "Cbz": "[N]-C(=O)-O-[CH2]-c1ccccc1",  # carbobenzoxy
            "Fmoc": "[N]-C(=O)-O-[CH2]-C1c2ccccc2-c2ccccc12"  # fluorenylmethoxycarbonyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track nitrogen atom mappings and their protecting group states
        nitrogen_states = self.track_nitrogen_protecting_groups(reactions)
        
        # Check if any nitrogen undergoes the specified cycling pattern
        cycling_detected = self.detect_cycling_pattern(nitrogen_states)
        
        return cycling_detected, len(reactions)
    
    def track_nitrogen_protecting_groups(self, reactions):
        """Track the protecting group state of each nitrogen atom through the route."""
        nitrogen_states = {}
        
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if r]
            
            if not prod_mol or not all(react_mols):
                continue
            
            # Get nitrogen atoms with map numbers from products
            prod_nitrogens = {}
            for atom in prod_mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetAtomMapNum() > 0:
                    map_num = atom.GetAtomMapNum()
                    pg_state = self.identify_protecting_group(prod_mol, atom)
                    prod_nitrogens[map_num] = pg_state
            
            # Get nitrogen atoms with map numbers from reactants
            react_nitrogens = {}
            for react_mol in react_mols:
                for atom in react_mol.GetAtoms():
                    if atom.GetSymbol() == 'N' and atom.GetAtomMapNum() > 0:
                        map_num = atom.GetAtomMapNum()
                        pg_state = self.identify_protecting_group(react_mol, atom)
                        react_nitrogens[map_num] = pg_state
            
            # Update nitrogen states based on changes
            for map_num in prod_nitrogens:
                if map_num not in nitrogen_states:
                    nitrogen_states[map_num] = []
                
                # Add reactant state if this is the first time seeing this nitrogen
                if len(nitrogen_states[map_num]) == 0 and map_num in react_nitrogens:
                    nitrogen_states[map_num].append(react_nitrogens[map_num])
                
                # Add product state
                nitrogen_states[map_num].append(prod_nitrogens[map_num])
        
        return nitrogen_states
    
    def identify_protecting_group(self, mol, nitrogen_atom):
        """Identify the protecting group on a nitrogen atom."""
        for group_name, pattern in self.protecting_group_patterns.items():
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                matches = mol.GetSubstructMatches(pattern_mol)
                for match in matches:
                    if nitrogen_atom.GetIdx() in match:
                        return group_name
        
        # Check if nitrogen is free (no protecting group)
        if nitrogen_atom.GetTotalNumHs() > 0 or nitrogen_atom.GetFormalCharge() != 0:
            return "H"
        
        return "unknown"
    
    def detect_cycling_pattern(self, nitrogen_states):
        """Detect if any nitrogen follows the specified cycling pattern."""
        for map_num, states in nitrogen_states.items():
            if len(states) < self.step_count:
                continue
            
            # Check if the sequence matches the expected protecting group cycle
            if len(self.groups) >= self.step_count:
                expected_sequence = self.groups[:self.step_count]
                
                # Check if any consecutive subsequence matches the expected pattern
                for i in range(len(states) - self.step_count + 1):
                    subsequence = states[i:i + self.step_count]
                    if subsequence == expected_sequence:
                        return True
                
                # Also check for the exact sequence anywhere in the states
                if states == expected_sequence or expected_sequence == states[-self.step_count:]:
                    return True
        
        return False
