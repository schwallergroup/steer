"""Generated evaluation code for: Sequential protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses a sequential protecting group strategy
    with specified groups (THP, TBDMS) in an orthogonal manner.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "sequential")
        self.groups = config.get("groups", ["THP", "TBDMS"])
        self.selectivity = config.get("selectivity", "orthogonal")
        
        # Define SMARTS patterns for protecting groups
        self.protecting_patterns = {
            "THP": "[CH]1O[CH2][CH2][CH2][CH2]1",  # Tetrahydropyranyl
            "TBDMS": "[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]",  # tert-Butyldimethylsilyl
            "Boc": "[C](=O)O[C]([CH3])([CH3])[CH3]",  # tert-Butoxycarbonyl
            "Bn": "[CH2]c1ccccc1",  # Benzyl
            "Ac": "[C](=O)[CH3]"  # Acetyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection and deprotection events
        protection_events = []
        deprotection_events = []
        
        for i, rxn in enumerate(reactions):
            for group in self.groups:
                if self.detect_protection(rxn, group):
                    protection_events.append((i, group, "protect"))
                elif self.detect_deprotection(rxn, group):
                    deprotection_events.append((i, group, "deprotect"))
        
        # Check if strategy meets criteria
        condition = self.evaluate_strategy(protection_events, deprotection_events, len(reactions))
        
        return condition, len(reactions)
    
    def detect_protection(self, rxn, group):
        """Detect if a protecting group is being installed in the reaction"""
        if group not in self.protecting_patterns:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Count protecting group motifs in reactants vs products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if None in reactant_mols or None in product_mols:
                return False
            
            pattern = Chem.MolFromSmarts(self.protecting_patterns[group])
            if pattern is None:
                return False
            
            reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in reactant_mols if mol)
            product_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in product_mols if mol)
            
            # Protection: more protecting groups in products than reactants
            return product_matches > reactant_matches
            
        except:
            return False
    
    def detect_deprotection(self, rxn, group):
        """Detect if a protecting group is being removed in the reaction"""
        if group not in self.protecting_patterns:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Count protecting group motifs in reactants vs products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if None in reactant_mols or None in product_mols:
                return False
            
            pattern = Chem.MolFromSmarts(self.protecting_patterns[group])
            if pattern is None:
                return False
            
            reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in reactant_mols if mol)
            product_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in product_mols if mol)
            
            # Deprotection: fewer protecting groups in products than reactants
            return reactant_matches > product_matches
            
        except:
            return False
    
    def evaluate_strategy(self, protection_events, deprotection_events, total_steps):
        """Evaluate if the protecting group strategy meets the specified criteria"""
        if not protection_events:
            return False
        
        # Check if all specified groups are used
        groups_used = set(event[1] for event in protection_events)
        if not all(group in groups_used for group in self.groups):
            return False
        
        if self.strategy_type == "sequential":
            # For sequential strategy, check temporal ordering
            if len(self.groups) >= 2:
                first_group_protection = min([event[0] for event in protection_events if event[1] == self.groups[0]], default=float('inf'))
                second_group_protection = min([event[0] for event in protection_events if event[1] == self.groups[1]], default=float('inf'))
                
                # Second group should be protected after first group
                if second_group_protection <= first_group_protection:
                    return False
        
        if self.selectivity == "orthogonal":
            # Check for orthogonal deprotection (selective removal)
            deprotection_groups = [event[1] for event in deprotection_events]
            if len(set(deprotection_groups)) >= 2:
                # Ensure deprotections happen at different stages
                deprotection_steps = {}
                for event in deprotection_events:
                    group = event[1]
                    step = event[0]
                    if group not in deprotection_steps:
                        deprotection_steps[group] = []
                    deprotection_steps[group].append(step)
                
                # Check if deprotections are temporally separated
                if len(deprotection_steps) >= 2:
                    step_ranges = []
                    for group_steps in deprotection_steps.values():
                        step_ranges.append((min(group_steps), max(group_steps)))
                    
                    # Ensure non-overlapping deprotection windows
                    step_ranges.sort()
                    for i in range(len(step_ranges) - 1):
                        if step_ranges[i][1] >= step_ranges[i+1][0]:
                            return False
        
        return True
