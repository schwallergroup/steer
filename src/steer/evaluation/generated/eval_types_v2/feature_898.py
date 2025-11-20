"""Generated evaluation code for: Extensive protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ExtensiveProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for extensive protecting group cycling strategies.
    Checks for multiple cycles of protection/deprotection of the same functional groups.
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("protection_deprotection_cycles", 3)
        self.target_groups = config.get("groups_involved", ["Boc", "Cbz", "acetate"])
        self.same_group_reprotected = config.get("same_functional_group_reprotected", True)
        
        # Define SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "Boc": "[NX3][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # Boc group
            "Cbz": "[NX3][CX3](=[OX1])[OX2][CH2][c1ccccc1]",  # Cbz group
            "acetate": "[OX2][CX3](=[OX1])[CH3]",  # Acetate ester
            "TBDMS": "[OX2][Si]([CH3])([CH3])[CX4]([CH3])([CH3])[CH3]",  # TBDMS
            "TMS": "[OX2][Si]([CH3])([CH3])[CH3]",  # TMS
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events for each group type
        pg_events = {group: [] for group in self.target_groups}
        
        for i, rxn in enumerate(reactions):
            for group in self.target_groups:
                if group in self.protecting_group_patterns:
                    event_type = self.analyze_protecting_group_change(rxn, group)
                    if event_type:
                        pg_events[group].append((i, event_type))
        
        # Count cycles for each protecting group
        total_cycles = 0
        groups_with_cycling = 0
        
        for group, events in pg_events.items():
            cycles = self.count_protection_cycles(events)
            if cycles > 0:
                total_cycles += cycles
                groups_with_cycling += 1
        
        # Check if we meet the cycling criteria
        meets_min_cycles = total_cycles >= self.min_cycles
        has_target_groups = any(group in pg_events and len(pg_events[group]) > 0 
                               for group in self.target_groups)
        
        if self.same_group_reprotected:
            # At least one group must have multiple protection/deprotection cycles
            has_reprotection = any(self.count_protection_cycles(events) >= 2 
                                 for events in pg_events.values())
            condition = meets_min_cycles and has_target_groups and has_reprotection
        else:
            condition = meets_min_cycles and has_target_groups
        
        return condition, len(reactions)

    def analyze_protecting_group_change(self, rxn, group_type):
        """
        Analyze if a reaction involves protection or deprotection of a specific group.
        Returns 'protect', 'deprotect', or None.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products_smiles.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return None
            
            pattern = Chem.MolFromSmarts(self.protecting_group_patterns[group_type])
            if pattern is None:
                return None
            
            # Count protecting groups in reactants and products
            reactant_pg_count = sum(len(mol.GetSubstructMatches(pattern)) 
                                  for mol in reactant_mols)
            product_pg_count = sum(len(mol.GetSubstructMatches(pattern)) 
                                 for mol in product_mols)
            
            if product_pg_count > reactant_pg_count:
                return 'protect'
            elif reactant_pg_count > product_pg_count:
                return 'deprotect'
            else:
                return None
                
        except Exception:
            return None

    def count_protection_cycles(self, events):
        """
        Count the number of protection/deprotection cycles from a list of events.
        A cycle is defined as protect -> deprotect sequence.
        """
        if len(events) < 2:
            return 0
            
        cycles = 0
        i = 0
        
        while i < len(events) - 1:
            current_event = events[i][1]
            
            if current_event == 'protect':
                # Look for next deprotection
                for j in range(i + 1, len(events)):
                    if events[j][1] == 'deprotect':
                        cycles += 1
                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1
                
        return cycles

    def route_scoring(self, x) -> float:
        """Convert condition result to score. Higher scores for more extensive cycling."""
        if x < 0:
            return 0  # No cycling detected
        else:
            # Score based on how early extensive cycling appears
            return max(0, 10 - x * 10)
