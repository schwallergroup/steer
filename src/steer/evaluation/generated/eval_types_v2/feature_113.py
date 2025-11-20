"""Generated evaluation code for: Protecting group cycling on alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes based on protecting group cycling on alcohol functional groups.
    Detects if the same alcohol undergoes protection/deprotection cycles with
    different protecting groups (TBS and benzyl) in the specified sequence.
    """
    
    def __init__(self, config):
        self.functional_group = config["functional_group"]
        self.protection_sequence = config["protection_sequence"]
        self.cycle_count = config["cycle_count"]
        
        # Define SMARTS patterns for protecting groups
        self.protecting_patterns = {
            "TBS": "[OH1][Si](C)(C)C(C)(C)C",  # TBS-protected alcohol
            "Bn": "[OH1]Cc1ccccc1",  # Benzyl-protected alcohol
            "free": "[OH1][C,c]"  # Free alcohol
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the protection sequence cycling occurs in the route."""
        reactions = self.get_rxns(d)
        
        # Track alcohol atoms through the route using atom mapping
        alcohol_states = self.track_alcohol_protection_states(reactions)
        
        # Check if any alcohol follows the specified protection sequence
        condition_met = False
        for alcohol_map_num, states in alcohol_states.items():
            if self.matches_protection_sequence(states):
                condition_met = True
                break
        
        return condition_met, len(reactions)
    
    def track_alcohol_protection_states(self, reactions):
        """Track the protection states of alcohol atoms through all reactions."""
        alcohol_states = {}
        
        for rxn_idx, rxn in enumerate(reactions):
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            try:
                prod_mols = [Chem.MolFromSmiles(products)]
                react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            except:
                continue
            
            # Find alcohol atoms and their states in products and reactants
            for mol in prod_mols + react_mols:
                if mol is None:
                    continue
                    
                for atom in mol.GetAtoms():
                    map_num = atom.GetAtomMapNum()
                    if map_num > 0 and atom.GetSymbol() == 'O':
                        state = self.determine_protection_state(mol, atom)
                        if state:
                            if map_num not in alcohol_states:
                                alcohol_states[map_num] = []
                            alcohol_states[map_num].append((rxn_idx, state))
        
        # Sort states by reaction index for each alcohol
        for map_num in alcohol_states:
            alcohol_states[map_num].sort(key=lambda x: x[0])
            alcohol_states[map_num] = [state for _, state in alcohol_states[map_num]]
        
        return alcohol_states
    
    def determine_protection_state(self, mol, oxygen_atom):
        """Determine the protection state of an alcohol oxygen atom."""
        atom_idx = oxygen_atom.GetIdx()
        
        # Check each protection pattern
        for state, pattern in self.protecting_patterns.items():
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol is None:
                continue
                
            matches = mol.GetSubstructMatches(patt_mol)
            for match in matches:
                if atom_idx in match:
                    return state
        
        return None
    
    def matches_protection_sequence(self, states):
        """Check if the observed states match the expected protection sequence."""
        if len(states) < len(self.protection_sequence):
            return False
        
        # Look for the protection sequence pattern
        sequence_matches = 0
        i = 0
        
        while i <= len(states) - len(self.protection_sequence):
            match = True
            for j, expected_state in enumerate(self.protection_sequence):
                if i + j >= len(states) or states[i + j] != expected_state:
                    match = False
                    break
            
            if match:
                sequence_matches += 1
                i += len(self.protection_sequence)
                
                # Check if we've found enough cycles
                if sequence_matches >= self.cycle_count:
                    return True
            else:
                i += 1
        
        return False
