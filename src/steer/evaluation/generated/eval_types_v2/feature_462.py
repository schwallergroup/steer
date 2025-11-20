"""Generated evaluation code for: Protecting group cycling with two different groups"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Checks for protecting group cycling with two different groups at the same position.
    Detects sequences where one protecting group is removed and replaced with another
    at the same atom position.
    """
    
    def __init__(self, config):
        self.protection_cycles = config["protection_cycles"]
        self.different_groups = config["different_groups"]
        self.same_position = config["same_position"]
        
        # SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "Cbz": "[N:1]C(=O)O[CH2]c1ccccc1",  # Cbz-protected nitrogen
            "Boc": "[N:1]C(=O)OC(C)(C)C",        # Boc-protected nitrogen
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group events
        protection_events = []
        
        for i, rxn in enumerate(reactions):
            event = self.analyze_protection_event(rxn)
            if event:
                protection_events.append((i, event))
        
        # Check if we have the required cycling pattern
        cycling_found = self.detect_cycling_pattern(protection_events)
        
        return cycling_found, len(reactions)
    
    def analyze_protection_event(self, rxn):
        """Analyze a reaction to detect protection/deprotection events."""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return None
        
        # Check for deprotection (protecting group present in reactants, absent in products)
        for group_name, pattern in self.protecting_group_patterns.items():
            if group_name in self.different_groups:
                pattern_mol = Chem.MolFromSmarts(pattern)
                
                # Find protected nitrogens in reactants
                protected_atoms = []
                for mol in reactants:
                    matches = mol.GetSubstructMatches(pattern_mol)
                    for match in matches:
                        # Get atom map number of the nitrogen (first atom in pattern)
                        n_atom = mol.GetAtomWithIdx(match[0])
                        if n_atom.GetAtomMapNum() > 0:
                            protected_atoms.append(n_atom.GetAtomMapNum())
                
                # Check if these nitrogens are deprotected in products
                for mol in products:
                    for atom_map in protected_atoms:
                        atom = self.get_atom_by_map(mol, atom_map)
                        if atom and not self.is_nitrogen_protected(mol, atom, pattern_mol):
                            return {"type": "deprotection", "group": group_name, "atom_map": atom_map}
                
                # Check for protection (free nitrogen in reactants, protected in products)
                free_nitrogens = []
                for mol in reactants:
                    for atom in mol.GetAtoms():
                        if (atom.GetSymbol() == "N" and atom.GetAtomMapNum() > 0 and
                            not self.is_nitrogen_protected(mol, atom, pattern_mol)):
                            free_nitrogens.append(atom.GetAtomMapNum())
                
                for mol in products:
                    matches = mol.GetSubstructMatches(pattern_mol)
                    for match in matches:
                        n_atom = mol.GetAtomWithIdx(match[0])
                        if n_atom.GetAtomMapNum() in free_nitrogens:
                            return {"type": "protection", "group": group_name, "atom_map": n_atom.GetAtomMapNum()}
        
        return None
    
    def detect_cycling_pattern(self, protection_events):
        """Detect if the protection events form the required cycling pattern."""
        if len(protection_events) < 2:
            return False
        
        # Group events by atom position if same_position is required
        if self.same_position:
            position_events = {}
            for step, event in protection_events:
                atom_map = event["atom_map"]
                if atom_map not in position_events:
                    position_events[atom_map] = []
                position_events[atom_map].append((step, event))
            
            # Check each position for cycling
            for atom_map, events in position_events.items():
                if self.check_cycling_at_position(events):
                    return True
            return False
        else:
            # Check overall cycling pattern
            return self.check_cycling_at_position(protection_events)
    
    def check_cycling_at_position(self, events):
        """Check if events at a position show the required cycling pattern."""
        # Sort by reaction step
        events = sorted(events, key=lambda x: x[0])
        
        # Look for alternating protection/deprotection with different groups
        group_sequence = []
        protection_state = None
        
        for step, event in events:
            if event["type"] == "deprotection":
                if protection_state == event["group"]:
                    group_sequence.append(("deprotect", event["group"]))
                    protection_state = None
            elif event["type"] == "protection":
                if protection_state is None:
                    group_sequence.append(("protect", event["group"]))
                    protection_state = event["group"]
        
        # Check if we have at least one complete cycle with different groups
        cycles_found = 0
        used_groups = set()
        
        i = 0
        while i < len(group_sequence) - 1:
            if (group_sequence[i][0] == "deprotect" and 
                group_sequence[i + 1][0] == "protect" and
                group_sequence[i][1] != group_sequence[i + 1][1]):
                
                used_groups.add(group_sequence[i][1])
                used_groups.add(group_sequence[i + 1][1])
                cycles_found += 1
                i += 2
            else:
                i += 1
        
        return (cycles_found >= self.protection_cycles and 
                len(used_groups.intersection(set(self.different_groups))) >= len(self.different_groups))
    
    def get_atom_by_map(self, mol, atom_map):
        """Get atom by its atom map number."""
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == atom_map:
                return atom
        return None
    
    def is_nitrogen_protected(self, mol, n_atom, pattern_mol):
        """Check if a nitrogen atom is protected by matching the pattern."""
        matches = mol.GetSubstructMatches(pattern_mol)
        for match in matches:
            if match[0] == n_atom.GetIdx():
                return True
        return False
