"""Generated evaluation code for: Sequential protecting group strategy for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential protecting group strategies on specific functional groups.
    Checks if the specified protecting groups are applied in sequence to the target functional group.
    """
    
    def __init__(self, config):
        self.functional_group = config["functional_group"]
        self.protection_count = config["protection_count"]
        self.protecting_groups = config["protecting_groups"]
        
        # Define SMARTS patterns for functional groups
        self.fg_patterns = {
            "amine": "[NX3;H2,H1;!$(NC=O)]",  # Primary or secondary amine, not amide
            "alcohol": "[OX2H]",
            "carboxylic_acid": "[CX3](=O)[OX2H1]"
        }
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[NX3][CX3](=O)[OX2][CX4]([CH3])([CH3])[CH3]",  # Boc-protected amine
            "formyl": "[NX3][CX3H1]=O",  # Formyl-protected amine
            "Cbz": "[NX3][CX3](=O)[OX2][CH2][c1ccccc1]",  # Cbz-protected amine
            "acetyl": "[NX3][CX3](=O)[CH3]",  # Acetyl-protected amine
            "TBS": "[OX2][SiX4]([CH3])([CH3])[CX4]([CH3])([CH3])[CH3]",  # TBS-protected alcohol
            "methyl": "[CX3](=O)[OX2][CH3]"  # Methyl ester (carboxylic acid protection)
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection events in order
        protection_events = []
        
        for i, rxn in enumerate(reactions):
            protection_type = self.detect_protection_reaction(rxn)
            if protection_type:
                protection_events.append((i, protection_type))
        
        # Check if we have the required number of protection events
        if len(protection_events) < self.protection_count:
            return False, len(reactions)
        
        # Check if the protecting groups match the specified sequence
        sequential_match = self.check_sequential_protection(protection_events)
        
        return sequential_match, len(reactions)
    
    def detect_protection_reaction(self, rxn):
        """
        Detect if a reaction involves protection of the target functional group
        Returns the protecting group type if found, None otherwise
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants = [Chem.MolFromSmiles(smi) for smi in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return None
            
            # Check if reactant has unprotected functional group
            fg_pattern = Chem.MolFromSmarts(self.fg_patterns.get(self.functional_group, ""))
            if not fg_pattern:
                return None
            
            has_unprotected_fg = any(mol.HasSubstructMatch(fg_pattern) for mol in reactants)
            
            if has_unprotected_fg:
                # Check which protecting group appears in products
                for pg_name, pg_pattern_smarts in self.pg_patterns.items():
                    pg_pattern = Chem.MolFromSmarts(pg_pattern_smarts)
                    if pg_pattern and any(mol.HasSubstructMatch(pg_pattern) for mol in products):
                        return pg_name
            
            return None
            
        except Exception:
            return None
    
    def check_sequential_protection(self, protection_events):
        """
        Check if protection events match the specified sequence
        """
        if len(protection_events) < len(self.protecting_groups):
            return False
        
        # Sort events by reaction order (depth in synthesis)
        protection_events.sort(key=lambda x: x[0])
        
        # Check if the first N protection events match our required sequence
        for i, required_pg in enumerate(self.protecting_groups):
            if i >= len(protection_events):
                return False
            
            _, detected_pg = protection_events[i]
            if detected_pg != required_pg:
                return False
        
        return True
    
    def route_scoring(self, x):
        """
        Score based on whether the sequential protection strategy is found
        """
        if x < 0:
            return 0  # Strategy not found
        else:
            return 1  # Strategy found (binary scoring for this feature)
