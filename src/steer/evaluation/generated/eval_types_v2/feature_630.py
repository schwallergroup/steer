"""Generated evaluation code for: Sequential protecting group cycling on hydroxyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for sequential protecting group cycling on hydroxyl groups.
    Detects patterns where a hydroxyl group is protected, then deprotected, 
    then modified in sequence.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "hydroxyl")
        self.protection_count = config.get("protection_count", 2)
        self.sequential = config.get("sequential", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < 3:  # Need at least protect, deprotect, modify
            return False, len(reactions)
        
        # Track hydroxyl modifications through the sequence
        protection_events = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_hydroxyl_protection(rxn):
                protection_events.append(('protect', i))
            elif self.detect_hydroxyl_deprotection(rxn):
                protection_events.append(('deprotect', i))
            elif self.detect_hydroxyl_modification(rxn):
                protection_events.append(('modify', i))
        
        # Check for sequential cycling pattern
        condition = self.check_sequential_cycling(protection_events)
        
        return condition, len(reactions)
    
    def detect_hydroxyl_protection(self, rxn):
        """Detect hydroxyl protection reactions (OH -> OR)"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Common protecting group patterns
        protected_patterns = [
            "[OH1][C](=O)[CH3]",  # Acetate
            "[OH1][Si]",          # Silyl ethers (TBS, TIPS, etc.)
            "[OH1][C]([CH3])([CH3])[C]",  # tert-butyl
            "[OH1][CH2][c1ccccc1]"        # Benzyl
        ]
        
        for pattern in protected_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol:
                # Check if product has protected OH that wasn't in reactants
                if prod_mol and prod_mol.HasSubstructMatch(pattern_mol):
                    for react_mol in react_mols:
                        if react_mol and not react_mol.HasSubstructMatch(pattern_mol):
                            # Check if reactant had free OH
                            oh_pattern = Chem.MolFromSmarts("[OH1]")
                            if oh_pattern and react_mol.HasSubstructMatch(oh_pattern):
                                return True
        return False
    
    def detect_hydroxyl_deprotection(self, rxn):
        """Detect hydroxyl deprotection reactions (OR -> OH)"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Look for free OH in product that was protected in reactant
        oh_pattern = Chem.MolFromSmarts("[OH1]")
        protected_patterns = [
            "[OH1][C](=O)[CH3]",  # Acetate
            "[OH1][Si]",          # Silyl ethers
            "[OH1][C]([CH3])([CH3])[C]",  # tert-butyl
            "[OH1][CH2][c1ccccc1]"        # Benzyl
        ]
        
        if prod_mol and oh_pattern and prod_mol.HasSubstructMatch(oh_pattern):
            for react_mol in react_mols:
                if react_mol:
                    for pattern in protected_patterns:
                        pattern_mol = Chem.MolFromSmarts(pattern)
                        if pattern_mol and react_mol.HasSubstructMatch(pattern_mol):
                            # Reactant was protected, product has free OH
                            return True
        return False
    
    def detect_hydroxyl_modification(self, rxn):
        """Detect modifications to hydroxyl groups (OH -> other functional group)"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        oh_pattern = Chem.MolFromSmarts("[OH1]")
        
        # Check if reactant had OH but product doesn't (or has different substitution)
        for react_mol in react_mols:
            if react_mol and oh_pattern and react_mol.HasSubstructMatch(oh_pattern):
                if prod_mol:
                    # Check for common OH modifications
                    ether_pattern = Chem.MolFromSmarts("[O]([C])[C]")  # Ether formation
                    ester_pattern = Chem.MolFromSmarts("[O][C](=O)")   # Ester formation
                    
                    if ((ether_pattern and prod_mol.HasSubstructMatch(ether_pattern)) or 
                        (ester_pattern and prod_mol.HasSubstructMatch(ester_pattern))):
                        return True
        return False
    
    def check_sequential_cycling(self, protection_events):
        """Check if protection events follow the expected cycling pattern"""
        if len(protection_events) < self.protection_count + 1:
            return False
        
        # Look for pattern: protect -> deprotect -> modify (or similar cycles)
        cycle_count = 0
        i = 0
        
        while i < len(protection_events) - 2:
            event_type, _ = protection_events[i]
            
            if event_type == 'protect':
                # Look for corresponding deprotect
                for j in range(i + 1, len(protection_events)):
                    next_event, _ = protection_events[j]
                    if next_event == 'deprotect':
                        # Look for modification after deprotection
                        for k in range(j + 1, len(protection_events)):
                            final_event, _ = protection_events[k]
                            if final_event in ['modify', 'protect']:
                                cycle_count += 1
                                i = k
                                break
                        else:
                            i += 1
                        break
                else:
                    i += 1
            else:
                i += 1
        
        return cycle_count >= self.protection_count
    
    def parse_reaction(self, rxn):
        """Parse reaction SMILES to get product and reactant molecules"""
        try:
            parts = rxn.split(">>")
            if len(parts) != 2:
                return None, []
            
            prod_mol = Chem.MolFromSmiles(parts[0])
            react_smiles = parts[1].split(".")
            react_mols = [Chem.MolFromSmiles(smi) for smi in react_smiles if smi.strip()]
            
            return prod_mol, [mol for mol in react_mols if mol is not None]
        except:
            return None, []
