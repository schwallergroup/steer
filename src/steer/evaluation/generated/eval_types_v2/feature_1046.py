"""Generated evaluation code for: Dual ester protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualEsterProtectingGroupCycling(MultiRxnCondBase):
    """
    Checks if a route uses a dual ester protecting group cycling strategy.
    The route should cycle between t-butyl ester, free acid, and methyl ester
    protections for the same carboxylic acid functionality.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.functional_group = config["parameters"]["functional_group"]
        self.cycle_count = config["parameters"]["cycle_count"]
        
        # SMARTS patterns for protecting groups
        self.tert_butyl_ester = "[CX3](=O)[OX2]C(C)(C)C"
        self.methyl_ester = "[CX3](=O)[OX2]C"
        self.carboxylic_acid = "[CX3](=O)[OX2H1]"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection state changes throughout the route
        protection_states = []
        
        for rxn in reactions:
            state_change = self.analyze_protection_state_change(rxn)
            if state_change:
                protection_states.append(state_change)
        
        # Check if we have the required cycling pattern
        has_cycling = self.detect_dual_ester_cycling(protection_states)
        
        return has_cycling, len(reactions)
    
    def analyze_protection_state_change(self, rxn):
        """Analyze if a reaction changes the protection state of carboxylic acid"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return None
        
        reactant_states = []
        product_states = []
        
        # Analyze reactants
        for mol in reactants:
            state = self.get_protection_state(mol)
            if state:
                reactant_states.extend(state)
        
        # Analyze products  
        for mol in products:
            state = self.get_protection_state(mol)
            if state:
                product_states.extend(state)
        
        # Check for protection state changes
        if len(reactant_states) == 1 and len(product_states) == 1:
            if reactant_states[0] != product_states[0]:
                return (reactant_states[0], product_states[0])
        
        return None
    
    def get_protection_state(self, mol):
        """Determine protection state of carboxylic acid groups in molecule"""
        states = []
        
        # Check for t-butyl ester
        if mol.HasSubstructMatch(Chem.MolFromSmarts(self.tert_butyl_ester)):
            states.append("tert_butyl_ester")
        
        # Check for methyl ester (but not t-butyl ester)
        if mol.HasSubstructMatch(Chem.MolFromSmarts(self.methyl_ester)) and \
           not mol.HasSubstructMatch(Chem.MolFromSmarts(self.tert_butyl_ester)):
            states.append("methyl_ester")
        
        # Check for free carboxylic acid
        if mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid)):
            states.append("carboxylic_acid")
        
        return states if states else None
    
    def detect_dual_ester_cycling(self, protection_states):
        """Check if protection states show dual ester cycling pattern"""
        if len(protection_states) < self.cycle_count:
            return False
        
        # Look for cycling between the specified protecting groups
        target_groups = set(self.protecting_groups + ["carboxylic_acid"])
        
        # Count transitions between target protection states
        valid_transitions = 0
        states_seen = set()
        
        for transition in protection_states:
            from_state, to_state = transition
            
            if from_state in target_groups and to_state in target_groups:
                valid_transitions += 1
                states_seen.add(from_state)
                states_seen.add(to_state)
        
        # Check if we have enough transitions and have seen all required states
        required_states = set(self.protecting_groups)
        has_required_cycling = (valid_transitions >= self.cycle_count and 
                               required_states.issubset(states_seen))
        
        return has_required_cycling
