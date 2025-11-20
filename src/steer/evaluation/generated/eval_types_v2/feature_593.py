"""Generated evaluation code for: Extensive protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on protecting group cycling strategy.
    Checks for multiple protect-deprotect cycles and total protection steps
    for specified functional groups.
    """
    
    def __init__(self, config):
        self.cycle_count = config.get("cycle_count", 2)
        self.functional_groups = config.get("functional_groups", ["alcohol", "carboxylic_acid"])
        self.total_protection_steps = config.get("total_protection_steps", 5)
        
        # Define SMARTS patterns for functional groups and their protecting groups
        self.fg_patterns = {
            "alcohol": {
                "free": "[OH1]",
                "protected": ["[O;R1]C1CCCCO1", "[O]C(=O)C", "[O]C(C)(C)C", "[O]Si", "[Cl]"]  # THP, acetate, tBu, silyl, Cl
            },
            "carboxylic_acid": {
                "free": "C(=O)[OH1]",
                "protected": ["C(=O)OC", "C(=O)OCC", "C(=O)OC(C)(C)C"]  # methyl, ethyl, tBu esters
            }
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events for each functional group
        fg_events = {fg: [] for fg in self.functional_groups}
        total_steps = 0
        
        for rxn in reactions:
            for fg in self.functional_groups:
                event = self.detect_protection_event(rxn, fg)
                if event:
                    fg_events[fg].append(event)
                    total_steps += 1
        
        # Count cycles for each functional group
        cycles_found = 0
        for fg in self.functional_groups:
            cycles_found += self.count_cycles(fg_events[fg])
        
        # Check if conditions are met
        condition = (cycles_found >= self.cycle_count and 
                    total_steps >= self.total_protection_steps)
        
        return condition, len(reactions)
    
    def detect_protection_event(self, rxn, functional_group):
        """Detect if a protection or deprotection event occurs for a functional group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not reactants or not products:
            return None
        
        patterns = self.fg_patterns[functional_group]
        free_pattern = Chem.MolFromSmarts(patterns["free"])
        protected_patterns = [Chem.MolFromSmarts(p) for p in patterns["protected"]]
        
        # Count free and protected groups in reactants and products
        reactant_free = len(reactants.GetSubstructMatches(free_pattern))
        reactant_protected = sum(len(reactants.GetSubstructMatches(p)) for p in protected_patterns if p)
        
        product_free = len(products.GetSubstructMatches(free_pattern))
        product_protected = sum(len(products.GetSubstructMatches(p)) for p in protected_patterns if p)
        
        # Determine event type
        if reactant_free > product_free and reactant_protected < product_protected:
            return "protect"
        elif reactant_free < product_free and reactant_protected > product_protected:
            return "deprotect"
        
        return None
    
    def count_cycles(self, events):
        """Count complete protect-deprotect cycles in the event sequence"""
        if len(events) < 2:
            return 0
        
        cycles = 0
        state = "free"  # Start assuming free state
        
        for event in events:
            if event == "protect" and state == "free":
                state = "protected"
            elif event == "deprotect" and state == "protected":
                state = "free"
                cycles += 1
        
        return cycles
