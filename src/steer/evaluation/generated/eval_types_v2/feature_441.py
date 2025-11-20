"""Generated evaluation code for: Multiple protection deprotection cycles same functional group"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectionDeprotectionCycles(MultiRxnCondBase):
    """
    Detects multiple protection-deprotection cycles on the same functional group.
    Tracks if a specific functional group undergoes multiple rounds of protection
    and deprotection with different protecting groups.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "alcohol")
        self.protection_cycles = config.get("protection_cycles", 2)
        self.protecting_groups = config.get("protecting_groups", ["acetate", "ethoxyethyl"])
        
        # Define SMARTS patterns for functional groups and protecting groups
        self.fg_patterns = {
            "alcohol": "[OH]",
            "amine": "[NH2,NH1]",
            "carboxylic_acid": "[CX3](=O)[OH]"
        }
        
        self.protection_patterns = {
            "acetate": "[OH]C(=O)C",
            "ethoxyethyl": "[OH]C([CH3])OCC",
            "silyl": "[OH][Si]",
            "benzyl": "[OH]Cc1ccccc1",
            "boc": "[NH]C(=O)OC(C)(C)C",
            "cbz": "[NH]C(=O)OCc1ccccc1"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events for each atom map number
        atom_protection_history = {}
        
        for rxn_idx, rxn in enumerate(reactions):
            protection_events = self.detect_protection_deprotection(rxn)
            
            for atom_map, event_type, protecting_group in protection_events:
                if atom_map not in atom_protection_history:
                    atom_protection_history[atom_map] = []
                atom_protection_history[atom_map].append({
                    'reaction': rxn_idx,
                    'type': event_type,
                    'protecting_group': protecting_group
                })
        
        # Check if any atom undergoes multiple protection-deprotection cycles
        condition_met = self.check_multiple_cycles(atom_protection_history)
        
        return condition_met, len(reactions)
    
    def detect_protection_deprotection(self, rxn):
        """
        Detect protection/deprotection events in a single reaction.
        Returns list of (atom_map_num, event_type, protecting_group) tuples.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
            
        reactants = [Chem.MolFromSmiles(smi) for smi in rxn_parts[0].split(".") if smi]
        products = [Chem.MolFromSmiles(smi) for smi in rxn_parts[1].split(".") if smi]
        
        if not all(reactants) or not all(products):
            return []
        
        events = []
        fg_pattern = Chem.MolFromSmarts(self.fg_patterns[self.functional_group])
        
        # Get atom map numbers for functional groups in reactants and products
        reactant_fg_atoms = set()
        product_fg_atoms = set()
        
        for mol in reactants:
            if mol.HasSubstructMatch(fg_pattern):
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0 and self.atom_matches_fg(atom, mol, fg_pattern):
                        reactant_fg_atoms.add(atom.GetAtomMapNum())
        
        for mol in products:
            if mol.HasSubstructMatch(fg_pattern):
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0 and self.atom_matches_fg(atom, mol, fg_pattern):
                        product_fg_atoms.add(atom.GetAtomMapNum())
        
        # Check for protection events (FG disappears, protected group appears)
        for mol in products:
            for pg_name in self.protecting_groups:
                if pg_name in self.protection_patterns:
                    pg_pattern = Chem.MolFromSmarts(self.protection_patterns[pg_name])
                    if mol.HasSubstructMatch(pg_pattern):
                        for atom in mol.GetAtoms():
                            map_num = atom.GetAtomMapNum()
                            if map_num > 0 and map_num in reactant_fg_atoms and map_num not in product_fg_atoms:
                                events.append((map_num, 'protection', pg_name))
        
        # Check for deprotection events (protected group disappears, FG appears)
        for mol in reactants:
            for pg_name in self.protecting_groups:
                if pg_name in self.protection_patterns:
                    pg_pattern = Chem.MolFromSmarts(self.protection_patterns[pg_name])
                    if mol.HasSubstructMatch(pg_pattern):
                        for atom in mol.GetAtoms():
                            map_num = atom.GetAtomMapNum()
                            if map_num > 0 and map_num not in reactant_fg_atoms and map_num in product_fg_atoms:
                                events.append((map_num, 'deprotection', pg_name))
        
        return events
    
    def atom_matches_fg(self, atom, mol, fg_pattern):
        """Check if an atom is part of the functional group pattern."""
        matches = mol.GetSubstructMatches(fg_pattern)
        atom_idx = atom.GetIdx()
        return any(atom_idx in match for match in matches)
    
    def check_multiple_cycles(self, atom_history):
        """
        Check if any atom undergoes the required number of protection-deprotection cycles
        with different protecting groups.
        """
        for atom_map, events in atom_history.items():
            if len(events) < self.protection_cycles * 2:  # Each cycle needs protection + deprotection
                continue
            
            # Sort events by reaction order
            events.sort(key=lambda x: x['reaction'])
            
            cycles_count = 0
            used_protecting_groups = set()
            expecting_deprotection = False
            current_pg = None
            
            for event in events:
                if event['type'] == 'protection' and not expecting_deprotection:
                    current_pg = event['protecting_group']
                    expecting_deprotection = True
                elif event['type'] == 'deprotection' and expecting_deprotection and event['protecting_group'] == current_pg:
                    cycles_count += 1
                    used_protecting_groups.add(current_pg)
                    expecting_deprotection = False
                    current_pg = None
            
            # Check if we have enough cycles with different protecting groups
            if cycles_count >= self.protection_cycles and len(used_protecting_groups) >= len(self.protecting_groups):
                return True
        
        return False
