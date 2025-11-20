"""Generated evaluation code for: Circular protection-deprotection sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CircularProtectionDeprotection(MultiRxnCondBase):
    """
    Detects circular protection-deprotection sequences where the same functional group
    is protected, deprotected, and re-protected with the same protecting group.
    """
    
    def __init__(self, config):
        self.protection_type = config.get("protection_type", "acetate")
        self.functional_group = config.get("functional_group", "alcohol")
        self.pattern_type = config.get("pattern", "circular")
        
        # Define SMARTS patterns for different protection types
        self.protection_patterns = {
            "acetate": "[OH1][CH1,CH2,CH3]",  # alcohol
            "boc": "[NH1,NH2][CH1,CH2,CH3]",  # amine
            "benzyl": "[OH1,NH1,NH2][CH1,CH2,CH3]"  # alcohol or amine
        }
        
        self.deprotection_patterns = {
            "acetate": "[OH1]C(=O)C",  # acetate ester
            "boc": "[NH1,NH2]C(=O)OC(C)(C)C",  # BOC carbamate
            "benzyl": "[OH1,NH1,NH2]Cc1ccccc1"  # benzyl ether/amine
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events by atom mapping
        protection_events = []
        deprotection_events = []
        
        for i, rxn in enumerate(reactions):
            if self.is_protection_reaction(rxn):
                protected_atoms = self.get_affected_atoms(rxn, "protection")
                protection_events.extend([(atom, i) for atom in protected_atoms])
            
            if self.is_deprotection_reaction(rxn):
                deprotected_atoms = self.get_affected_atoms(rxn, "deprotection")
                deprotection_events.extend([(atom, i) for atom in deprotected_atoms])
        
        # Check for circular pattern: same atom protected → deprotected → re-protected
        circular_found = self.detect_circular_pattern(protection_events, deprotection_events)
        
        return circular_found, len(reactions)
    
    def is_protection_reaction(self, rxn):
        """Check if reaction involves protection of functional group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not reactants or not products:
            return False
        
        # Check if free functional group in reactants becomes protected in products
        free_pattern = self.protection_patterns.get(self.protection_type)
        protected_pattern = self.deprotection_patterns.get(self.protection_type)
        
        if not free_pattern or not protected_pattern:
            return False
        
        has_free_reactant = reactants.HasSubstructMatch(Chem.MolFromSmarts(free_pattern))
        has_protected_product = products.HasSubstructMatch(Chem.MolFromSmarts(protected_pattern))
        
        return has_free_reactant and has_protected_product
    
    def is_deprotection_reaction(self, rxn):
        """Check if reaction involves deprotection of functional group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not reactants or not products:
            return False
        
        # Check if protected functional group in reactants becomes free in products
        free_pattern = self.protection_patterns.get(self.protection_type)
        protected_pattern = self.deprotection_patterns.get(self.protection_type)
        
        if not free_pattern or not protected_pattern:
            return False
        
        has_protected_reactant = reactants.HasSubstructMatch(Chem.MolFromSmarts(protected_pattern))
        has_free_product = products.HasSubstructMatch(Chem.MolFromSmarts(free_pattern))
        
        return has_protected_reactant and has_free_product
    
    def get_affected_atoms(self, rxn, reaction_type):
        """Extract atom map numbers of atoms involved in protection/deprotection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
        
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not reactants or not products:
            return []
        
        affected_atoms = []
        
        # Get atoms involved in the functional group change
        if reaction_type == "protection":
            pattern = self.protection_patterns.get(self.protection_type)
            mol = reactants
        else:  # deprotection
            pattern = self.deprotection_patterns.get(self.protection_type)
            mol = reactants
        
        if pattern:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetAtomMapNum() > 0:
                        affected_atoms.append(atom.GetAtomMapNum())
        
        return affected_atoms
    
    def detect_circular_pattern(self, protection_events, deprotection_events):
        """Detect if the same atom undergoes protection → deprotection → re-protection"""
        # Group events by atom map number
        atom_events = {}
        
        for atom, step in protection_events:
            if atom not in atom_events:
                atom_events[atom] = []
            atom_events[atom].append(('protect', step))
        
        for atom, step in deprotection_events:
            if atom not in atom_events:
                atom_events[atom] = []
            atom_events[atom].append(('deprotect', step))
        
        # Check each atom for circular pattern
        for atom, events in atom_events.items():
            # Sort events by reaction step
            events.sort(key=lambda x: x[1])
            event_types = [e[0] for e in events]
            
            # Look for pattern: protect → deprotect → protect (circular)
            if len(event_types) >= 3:
                for i in range(len(event_types) - 2):
                    if (event_types[i] == 'protect' and 
                        event_types[i + 1] == 'deprotect' and 
                        event_types[i + 2] == 'protect'):
                        return True
        
        return False
