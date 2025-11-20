"""Generated evaluation code for: Protecting group cycling with Cbz-Teoc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes based on protecting group cycling strategies.
    Checks if the route involves protecting with one group, deprotecting, 
    then re-protecting with a different group on the same atom type.
    """
    
    def __init__(self, config):
        self.protection_sequence = config["protection_sequence"]
        self.atom_type = config["atom_type"]
        self.cycle_count = config["cycle_count"]
        
        # Define protecting group patterns
        self.protecting_groups = {
            "Cbz": "[CH2]c1ccccc1",  # Benzyl carbamate
            "Teoc": "[CH2][CH2][Si]",  # 2-(trimethylsilyl)ethoxycarbonyl
            "Boc": "C(C)(C)(C)OC(=O)",  # tert-butoxycarbonyl
            "Fmoc": "c1ccc2c(c1)cc(c2)[CH2]"  # Fluorenylmethoxycarbonyl
        }
        
        # Atom type patterns
        self.atom_patterns = {
            "nitrogen": "[N]",
            "oxygen": "[O]",
            "sulfur": "[S]"
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group changes across reactions
        pg_events = []
        
        for i, rxn in enumerate(reactions):
            event = self.analyze_protecting_group_change(rxn)
            if event:
                pg_events.append((i, event))
        
        # Check if the protection sequence matches the desired pattern
        cycle_detected = self.detect_protection_cycling(pg_events)
        
        return cycle_detected, len(reactions)

    def analyze_protecting_group_change(self, rxn):
        """Analyze a reaction to detect protecting group addition/removal"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for protection (PG addition)
            for pg_name, pg_pattern in self.protecting_groups.items():
                pg_smarts = Chem.MolFromSmarts(pg_pattern)
                if pg_smarts is None:
                    continue
                    
                reactant_has_pg = any(mol.HasSubstructMatch(pg_smarts) for mol in reactant_mols)
                product_has_pg = any(mol.HasSubstructMatch(pg_smarts) for mol in product_mols)
                
                if not reactant_has_pg and product_has_pg:
                    # Protection reaction
                    if self.has_target_atom_type(product_mols):
                        return ("protect", pg_name)
                elif reactant_has_pg and not product_has_pg:
                    # Deprotection reaction
                    if self.has_target_atom_type(reactant_mols):
                        return ("deprotect", pg_name)
                        
        except Exception:
            pass
            
        return None

    def has_target_atom_type(self, mols):
        """Check if molecules contain the target atom type"""
        if self.atom_type not in self.atom_patterns:
            return False
            
        atom_pattern = self.atom_patterns[self.atom_type]
        atom_smarts = Chem.MolFromSmarts(atom_pattern)
        
        if atom_smarts is None:
            return False
            
        return any(mol.HasSubstructMatch(atom_smarts) for mol in mols if mol is not None)

    def detect_protection_cycling(self, pg_events):
        """Detect if the protecting group events match the desired sequence"""
        if len(pg_events) < len(self.protection_sequence):
            return False
            
        # Look for the sequence pattern
        sequence_matches = 0
        i = 0
        
        while i < len(pg_events) and sequence_matches < self.cycle_count:
            seq_index = 0
            temp_i = i
            
            # Try to match the full sequence starting from position i
            while (seq_index < len(self.protection_sequence) and 
                   temp_i < len(pg_events)):
                
                expected = self.protection_sequence[seq_index]
                actual_event = pg_events[temp_i][1]
                
                if expected == "deprotection":
                    if actual_event[0] == "deprotect":
                        seq_index += 1
                        temp_i += 1
                    else:
                        break
                elif expected in self.protecting_groups:
                    if (actual_event[0] == "protect" and 
                        actual_event[1] == expected):
                        seq_index += 1
                        temp_i += 1
                    else:
                        break
                else:
                    break
            
            # Check if we matched the complete sequence
            if seq_index == len(self.protection_sequence):
                sequence_matches += 1
                i = temp_i
            else:
                i += 1
        
        return sequence_matches >= self.cycle_count
