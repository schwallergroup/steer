"""Generated evaluation code for: Extensive protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes based on protecting group cycling strategy.
    Checks for multiple cycles of protection/deprotection on the same functional groups.
    """
    
    def __init__(self, config):
        self.target_cycles = config["protection_deprotection_cycles"]
        self.protecting_groups = config["protecting_groups"]
        self.strategy_type = config["strategy_type"]
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[N:1]C(=O)OC(C)(C)C",
            "Cbz": "[N:1]C(=O)OCc1ccccc1",
            "benzyl_ester": "[C:1](=O)OCc1ccccc1"
        }
        
        # Compile patterns for protected forms
        self.protected_patterns = {}
        for pg in self.protecting_groups:
            if pg in self.pg_patterns:
                self.protected_patterns[pg] = Chem.MolFromSmarts(self.pg_patterns[pg])

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group events for each functional group position
        pg_events = {}  # Maps atom_map_num to list of (depth, event_type, pg_type)
        
        for depth, rxn in enumerate(reactions):
            protection_events = self.detect_protection(rxn)
            deprotection_events = self.detect_deprotection(rxn)
            
            # Record protection events
            for atom_map, pg_type in protection_events:
                if atom_map not in pg_events:
                    pg_events[atom_map] = []
                pg_events[atom_map].append((depth, 'protect', pg_type))
            
            # Record deprotection events
            for atom_map, pg_type in deprotection_events:
                if atom_map not in pg_events:
                    pg_events[atom_map] = []
                pg_events[atom_map].append((depth, 'deprotect', pg_type))
        
        # Count cycles for each position
        max_cycles = 0
        for atom_map, events in pg_events.items():
            cycles = self.count_cycles(events)
            max_cycles = max(max_cycles, cycles)
        
        # Check if target cycles are met
        condition = max_cycles >= self.target_cycles
        return condition, len(reactions)

    def detect_protection(self, rxn):
        """Detect protection reactions and return list of (atom_map, pg_type)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
            
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        protections = []
        
        for pg_type, pattern in self.protected_patterns.items():
            if pattern is None:
                continue
                
            # Find protected atoms in products
            for prod in products:
                if prod and prod.HasSubstructMatch(pattern):
                    matches = prod.GetSubstructMatches(pattern)
                    for match in matches:
                        protected_atom = prod.GetAtomWithIdx(match[0])
                        atom_map = protected_atom.GetAtomMapNum()
                        
                        # Check if this atom was unprotected in reactants
                        if self.was_unprotected_in_reactants(atom_map, pg_type, reactants):
                            protections.append((atom_map, pg_type))
        
        return protections

    def detect_deprotection(self, rxn):
        """Detect deprotection reactions and return list of (atom_map, pg_type)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
            
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        deprotections = []
        
        for pg_type, pattern in self.protected_patterns.items():
            if pattern is None:
                continue
                
            # Find protected atoms in reactants
            for react in reactants:
                if react and react.HasSubstructMatch(pattern):
                    matches = react.GetSubstructMatches(pattern)
                    for match in matches:
                        protected_atom = react.GetAtomWithIdx(match[0])
                        atom_map = protected_atom.GetAtomMapNum()
                        
                        # Check if this atom is unprotected in products
                        if self.is_unprotected_in_products(atom_map, pg_type, products):
                            deprotections.append((atom_map, pg_type))
        
        return deprotections

    def was_unprotected_in_reactants(self, atom_map, pg_type, reactants):
        """Check if atom with given map number was unprotected in reactants"""
        pattern = self.protected_patterns[pg_type]
        if pattern is None:
            return True
            
        for react in reactants:
            if react:
                for atom in react.GetAtoms():
                    if atom.GetAtomMapNum() == atom_map:
                        # Check if this atom is part of a protected group
                        matches = react.GetSubstructMatches(pattern)
                        for match in matches:
                            if atom.GetIdx() in match:
                                return False
                        return True
        return True

    def is_unprotected_in_products(self, atom_map, pg_type, products):
        """Check if atom with given map number is unprotected in products"""
        pattern = self.protected_patterns[pg_type]
        if pattern is None:
            return True
            
        for prod in products:
            if prod:
                for atom in prod.GetAtoms():
                    if atom.GetAtomMapNum() == atom_map:
                        # Check if this atom is part of a protected group
                        matches = prod.GetSubstructMatches(pattern)
                        for match in matches:
                            if atom.GetIdx() in match:
                                return False
                        return True
        return True

    def count_cycles(self, events):
        """Count protection/deprotection cycles from list of events"""
        if len(events) < 2:
            return 0
            
        # Sort events by depth
        events.sort(key=lambda x: x[0])
        
        cycles = 0
        state = 'unprotected'  # Start assuming unprotected
        
        for depth, event_type, pg_type in events:
            if event_type == 'protect' and state == 'unprotected':
                state = 'protected'
            elif event_type == 'deprotect' and state == 'protected':
                state = 'unprotected'
                cycles += 1
                
        return cycles

    def route_scoring(self, x):
        """Convert condition result to score (0-10)"""
        if x < 0:
            return 0  # Condition not met
        else:
            # Higher score for meeting the cycling requirement
            return 10 - abs(x - self.target_cycles) * 2
