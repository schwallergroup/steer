"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multiple protecting group cycling strategies.
    Checks if the same position undergoes protection/deprotection cycles with
    different protecting groups.
    """
    
    def __init__(self, config):
        self.required_cycles = config.get("protection_deprotection_cycles", 2)
        self.same_position = config.get("same_position", True)
        
        # Common protecting group patterns
        self.protecting_groups = {
            "boc": "[CH3][CH3][CH3]OC(=O)",
            "cbz": "c1ccccc1[CH2]OC(=O)", 
            "fmoc": "c1ccc2c(c1)c(c3ccccc32)[CH2]OC(=O)",
            "sem": "[Si]([CH3])([CH3])[CH2][CH2]O[CH2]",
            "bn": "c1ccccc1[CH2]",
            "pmc": "c1cc(OC)c(C)c(OC)c1[CH2]OC(=O)",
            "alloc": "C=CC[CH2]OC(=O)",
            "tosyl": "c1ccc(cc1)[S](=O)(=O)",
            "mesyl": "[CH3][S](=O)(=O)"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events by atom mapping
        protection_events = []  # [(atom_map, pg_type, is_protection, rxn_idx)]
        
        for idx, rxn in enumerate(reactions):
            events = self.analyze_protection_deprotection(rxn, idx)
            protection_events.extend(events)
        
        # Check for cycling at same position
        if self.same_position:
            cycles = self.count_position_cycles(protection_events)
        else:
            cycles = self.count_total_cycles(protection_events)
        
        condition_met = cycles >= self.required_cycles
        return condition_met, len(reactions)
    
    def analyze_protection_deprotection(self, rxn, rxn_idx):
        """Analyze a single reaction for protection/deprotection events."""
        events = []
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return events
                
            reactants = [Chem.MolFromSmiles(s) for s in rxn_parts[0].split(".") if s]
            products = [Chem.MolFromSmiles(s) for s in rxn_parts[1].split(".") if s]
            
            if not all(reactants) or not all(products):
                return events
            
            # Get all mapped atoms
            reactant_maps = {}
            product_maps = {}
            
            for mol in reactants:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        reactant_maps[atom.GetAtomMapNum()] = mol
            
            for mol in products:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        product_maps[atom.GetAtomMapNum()] = mol
            
            # Check for protection (PG addition)
            for pg_name, pg_pattern in self.protecting_groups.items():
                pg_mol = Chem.MolFromSmarts(pg_pattern)
                if pg_mol is None:
                    continue
                
                # Protection: PG appears in products but not reactants
                for prod_mol in products:
                    if prod_mol.HasSubstructMatch(pg_mol):
                        # Find which mapped atom is involved
                        matches = prod_mol.GetSubstructMatches(pg_mol)
                        for match in matches:
                            for atom_idx in match:
                                atom = prod_mol.GetAtomWithIdx(atom_idx)
                                if atom.GetAtomMapNum() > 0:
                                    # Check if this PG wasn't in reactants
                                    pg_in_reactants = False
                                    if atom.GetAtomMapNum() in reactant_maps:
                                        react_mol = reactant_maps[atom.GetAtomMapNum()]
                                        if react_mol.HasSubstructMatch(pg_mol):
                                            pg_in_reactants = True
                                    
                                    if not pg_in_reactants:
                                        events.append((atom.GetAtomMapNum(), pg_name, True, rxn_idx))
                
                # Deprotection: PG disappears from reactants to products
                for react_mol in reactants:
                    if react_mol.HasSubstructMatch(pg_mol):
                        matches = react_mol.GetSubstructMatches(pg_mol)
                        for match in matches:
                            for atom_idx in match:
                                atom = react_mol.GetAtomWithIdx(atom_idx)
                                if atom.GetAtomMapNum() > 0:
                                    # Check if this PG is gone in products
                                    pg_in_products = False
                                    if atom.GetAtomMapNum() in product_maps:
                                        prod_mol = product_maps[atom.GetAtomMapNum()]
                                        if prod_mol.HasSubstructMatch(pg_mol):
                                            pg_in_products = True
                                    
                                    if not pg_in_products:
                                        events.append((atom.GetAtomMapNum(), pg_name, False, rxn_idx))
        
        except Exception:
            pass
        
        return events
    
    def count_position_cycles(self, events):
        """Count complete protection-deprotection cycles at the same position."""
        # Group events by atom position
        position_events = {}
        for atom_map, pg_type, is_protection, rxn_idx in events:
            if atom_map not in position_events:
                position_events[atom_map] = []
            position_events[atom_map].append((pg_type, is_protection, rxn_idx))
        
        max_cycles = 0
        for atom_map, atom_events in position_events.items():
            # Sort by reaction index
            atom_events.sort(key=lambda x: x[2])
            
            # Count cycles (protection followed by deprotection)
            cycles = 0
            protected_groups = set()
            
            for pg_type, is_protection, _ in atom_events:
                if is_protection:
                    protected_groups.add(pg_type)
                else:  # deprotection
                    if pg_type in protected_groups:
                        protected_groups.remove(pg_type)
                        cycles += 1
            
            max_cycles = max(max_cycles, cycles)
        
        return max_cycles
    
    def count_total_cycles(self, events):
        """Count total protection-deprotection cycles across all positions."""
        # Sort all events by reaction index
        events.sort(key=lambda x: x[3])
        
        cycles = 0
        protected_positions = {}  # {(atom_map, pg_type): True}
        
        for atom_map, pg_type, is_protection, _ in events:
            key = (atom_map, pg_type)
            
            if is_protection:
                protected_positions[key] = True
            else:  # deprotection
                if key in protected_positions:
                    del protected_positions[key]
                    cycles += 1
        
        return cycles
