"""Generated evaluation code for: Multiple protecting group strategy with sequential cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes that employ multiple protecting group strategies with sequential cycling.
    Checks for the presence of specified protecting groups (Boc, Cbz) and validates
    sequential protection-deprotection cycles on multiple nitrogen functionalities.
    """
    
    def __init__(self, config):
        self.pg_types = config.get("pg_types", ["Boc", "Cbz"])
        self.sequential_operations = config.get("sequential_operations", True)
        self.multiple_nitrogens = config.get("multiple_nitrogens", True)
        
        # SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[NX3][C](=[O])[O][C]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "Cbz": "[NX3][C](=[O])[O][CH2]c1ccccc1"  # benzyloxycarbonyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        pg_operations = []
        protection_events = []
        deprotection_events = []
        
        for i, rxn in enumerate(reactions):
            for pg_type in self.pg_types:
                if self.detect_protection(rxn, pg_type):
                    pg_operations.append((i, "protection", pg_type))
                    protection_events.append((i, pg_type))
                elif self.detect_deprotection(rxn, pg_type):
                    pg_operations.append((i, "deprotection", pg_type))
                    deprotection_events.append((i, pg_type))
        
        # Check if multiple PG types are used
        used_pg_types = set(op[2] for op in pg_operations)
        multiple_pg_types = len(used_pg_types.intersection(set(self.pg_types))) >= 2
        
        # Check sequential operations (protection followed by deprotection)
        has_sequential = False
        if self.sequential_operations:
            has_sequential = self.check_sequential_cycles(protection_events, deprotection_events)
        
        # Check multiple nitrogen handling
        has_multiple_n = False
        if self.multiple_nitrogens:
            has_multiple_n = self.check_multiple_nitrogen_handling(reactions, pg_operations)
        
        # Overall condition: must have multiple PG types and meet other requirements
        condition = (multiple_pg_types and 
                    (not self.sequential_operations or has_sequential) and
                    (not self.multiple_nitrogens or has_multiple_n))
        
        return condition, len(reactions)
    
    def detect_protection(self, rxn, pg_type):
        """Detect if a protecting group is being installed"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
        if not pattern:
            return False
        
        # PG absent in reactants but present in products
        reactant_has_pg = any(mol.HasSubstructMatch(pattern) for mol in reactants)
        product_has_pg = any(mol.HasSubstructMatch(pattern) for mol in products)
        
        return not reactant_has_pg and product_has_pg
    
    def detect_deprotection(self, rxn, pg_type):
        """Detect if a protecting group is being removed"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
        if not pattern:
            return False
        
        # PG present in reactants but absent in products
        reactant_has_pg = any(mol.HasSubstructMatch(pattern) for mol in reactants)
        product_has_pg = any(mol.HasSubstructMatch(pattern) for mol in products)
        
        return reactant_has_pg and not product_has_pg
    
    def check_sequential_cycles(self, protection_events, deprotection_events):
        """Check if there are protection-deprotection cycles"""
        for prot_step, prot_type in protection_events:
            for deprot_step, deprot_type in deprotection_events:
                if prot_type == deprot_type and prot_step < deprot_step:
                    return True
        return False
    
    def check_multiple_nitrogen_handling(self, reactions, pg_operations):
        """Check if multiple nitrogens are being managed with protecting groups"""
        # Look for reactions that have multiple nitrogen-containing molecules
        # or evidence of differential protection on multiple sites
        nitrogen_pattern = Chem.MolFromSmarts("[NX3,NX2]")
        
        multiple_n_reactions = 0
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
            
            if not all(reactants) or not all(products):
                continue
            
            # Count nitrogens in reactants and products
            for mol_list in [reactants, products]:
                for mol in mol_list:
                    n_count = len(mol.GetSubstructMatches(nitrogen_pattern))
                    if n_count >= 2:
                        multiple_n_reactions += 1
                        break
        
        # Also check if we have overlapping protection operations (evidence of multiple sites)
        pg_type_steps = {}
        for step, op_type, pg_type in pg_operations:
            if pg_type not in pg_type_steps:
                pg_type_steps[pg_type] = []
            pg_type_steps[pg_type].append((step, op_type))
        
        overlapping_protection = any(len(steps) > 2 for steps in pg_type_steps.values())
        
        return multiple_n_reactions > 0 or overlapping_protection
