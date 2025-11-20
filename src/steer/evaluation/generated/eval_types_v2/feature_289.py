"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on protecting group cycling strategy.
    Checks if the route involves multiple cycles of protection/deprotection
    with specific protecting groups (Cbz, Boc, SEM, benzyl).
    """
    
    def __init__(self, config):
        self.required_cycles = config["protection_deprotection_cycles"]
        self.target_groups = config["groups"]
        
        # SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Cbz": "[NH]C(=O)OCC1=CC=CC=C1",  # Carboxybenzyl
            "Boc": "[NH]C(=O)OC(C)(C)C",      # tert-Butoxycarbonyl
            "SEM": "[NH]C(=O)OCC[Si](C)(C)C", # 2-(Trimethylsilyl)ethoxymethyl
            "benzyl": "[NH]CC1=CC=CC=C1"      # Benzyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        cycles_found = self.count_protection_cycles(reactions)
        condition = cycles_found >= self.required_cycles
        return condition, len(reactions)
    
    def count_protection_cycles(self, reactions):
        """Count complete protection/deprotection cycles for target groups."""
        group_states = {}  # Track protection state for each group
        cycles = 0
        
        for rxn in reactions:
            for group in self.target_groups:
                if group not in group_states:
                    group_states[group] = {"protected": False, "cycles": 0}
                
                is_protection = self.detect_protection(rxn, group)
                is_deprotection = self.detect_deprotection(rxn, group)
                
                if is_protection and not group_states[group]["protected"]:
                    group_states[group]["protected"] = True
                elif is_deprotection and group_states[group]["protected"]:
                    group_states[group]["protected"] = False
                    group_states[group]["cycles"] += 1
        
        # Count total cycles across all groups
        total_cycles = sum(state["cycles"] for state in group_states.values())
        return total_cycles
    
    def detect_protection(self, rxn, group):
        """Detect if a reaction introduces a protecting group."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".") if r]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".") if p]
        
        if not all(reactants + products):
            return False
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[group])
        if not pattern:
            return False
        
        # Check if protecting group appears in products but not reactants
        reactant_matches = sum(1 for mol in reactants if mol.HasSubstructMatch(pattern))
        product_matches = sum(1 for mol in products if mol.HasSubstructMatch(pattern))
        
        return product_matches > reactant_matches
    
    def detect_deprotection(self, rxn, group):
        """Detect if a reaction removes a protecting group."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".") if r]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".") if p]
        
        if not all(reactants + products):
            return False
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[group])
        if not pattern:
            return False
        
        # Check if protecting group disappears from reactants to products
        reactant_matches = sum(1 for mol in reactants if mol.HasSubstructMatch(pattern))
        product_matches = sum(1 for mol in products if mol.HasSubstructMatch(pattern))
        
        return reactant_matches > product_matches
