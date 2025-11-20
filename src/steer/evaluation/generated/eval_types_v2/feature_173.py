"""Generated evaluation code for: Sequential protecting group cycling at 5' position"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects sequential protecting group cycling at the 5' position.
    Checks if DMT protection is removed and then acetate protection is added
    at the same hydroxyl position within the specified number of cycles.
    """
    
    def __init__(self, config):
        self.position = config["parameters"]["position"]
        self.sequence = config["parameters"]["sequence"]
        self.cycle_count = config["parameters"]["cycle_count"]
        
        # Define SMARTS patterns for protecting groups
        self.dmt_pattern = "[OH1][CH2][CH]1O[CH][CH]([OH1])[CH]1[OH1]"  # 5' hydroxyl with DMT
        self.acetate_pattern = "[OH0]([CH2][CH]1O[CH][CH]([OH1])[CH]1[OH1])C(=O)C"  # 5' acetate ester
        self.free_hydroxyl_pattern = "[OH1][CH2][CH]1O[CH][CH]([OH1])[CH]1[OH1]"  # Free 5' hydroxyl
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track the state transitions
        state_transitions = []
        
        for rxn in reactions:
            transition = self.detect_protection_state_change(rxn)
            if transition:
                state_transitions.append(transition)
        
        # Check if we have the required sequential cycling pattern
        cycles_detected = self.analyze_cycling_pattern(state_transitions)
        condition_met = cycles_detected >= self.cycle_count
        
        return condition_met, len(reactions)
    
    def detect_protection_state_change(self, rxn):
        """Detect what type of protecting group change occurred in this reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products_smiles = rxn_parts[1].split(".")
        products = [Chem.MolFromSmiles(p) for p in products_smiles if p]
        
        if not reactants or not products:
            return None
        
        # Check initial state
        initial_state = self.get_protection_state(reactants)
        
        # Check final state (look at main product, usually the largest)
        final_states = [self.get_protection_state(p) for p in products]
        final_state = None
        
        # Find the product that represents the main synthetic intermediate
        for state in final_states:
            if state != "unknown":
                final_state = state
                break
        
        if initial_state and final_state and initial_state != final_state:
            return (initial_state, final_state)
        
        return None
    
    def get_protection_state(self, mol):
        """Determine the protection state of the 5' position"""
        if not mol:
            return "unknown"
            
        dmt_mol = Chem.MolFromSmarts(self.dmt_pattern)
        acetate_mol = Chem.MolFromSmarts(self.acetate_pattern)
        free_mol = Chem.MolFromSmarts(self.free_hydroxyl_pattern)
        
        if dmt_mol and mol.HasSubstructMatch(dmt_mol):
            return "DMT"
        elif acetate_mol and mol.HasSubstructMatch(acetate_mol):
            return "acetate"
        elif free_mol and mol.HasSubstructMatch(free_mol):
            return "free"
        else:
            return "unknown"
    
    def analyze_cycling_pattern(self, transitions):
        """Analyze state transitions to count complete cycles"""
        if len(transitions) < len(self.sequence) - 1:
            return 0
        
        cycles = 0
        expected_sequence = self.sequence
        current_position = 0
        
        for transition in transitions:
            from_state, to_state = transition
            
            # Check if this transition matches the expected sequence step
            if current_position < len(expected_sequence) - 1:
                expected_from = expected_sequence[current_position]
                expected_to = expected_sequence[current_position + 1]
                
                if from_state == expected_from and to_state == expected_to:
                    current_position += 1
                    
                    # If we completed a full sequence, count as one cycle
                    if current_position == len(expected_sequence) - 1:
                        cycles += 1
                        current_position = 0  # Reset for next cycle
                else:
                    # Reset if transition doesn't match expected pattern
                    current_position = 0
        
        return cycles
