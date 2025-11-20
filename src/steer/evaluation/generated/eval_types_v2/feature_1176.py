"""Generated evaluation code for: Sequential dual protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialDualProtectingGroup(MultiRxnCondBase):
    """
    Evaluates routes for sequential dual protecting group strategy.
    Checks if multiple protecting groups are used in sequence on different nitrogens.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["trityl", "cbz", "dmb", "benzyl"])
        self.sequential_operations = config.get("sequential_operations", True)
        self.multiple_nitrogens = config.get("multiple_nitrogens", True)
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "trityl": "[CH0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
            "cbz": "C(=O)Oc1ccccc1",
            "dmb": "c1cc(OC)c(OC)cc1C",
            "benzyl": "Cc1ccccc1"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        protection_events = []
        deprotection_events = []
        
        for i, rxn in enumerate(reactions):
            pg_ops = self.detect_protecting_group_operations(rxn)
            for op_type, pg_type in pg_ops:
                if op_type == "protection":
                    protection_events.append((i, pg_type))
                elif op_type == "deprotection":
                    deprotection_events.append((i, pg_type))
        
        # Check if strategy meets criteria
        condition = self.evaluate_strategy(protection_events, deprotection_events, reactions)
        
        return condition, len(reactions)
    
    def detect_protecting_group_operations(self, rxn):
        """Detect protecting group installation and removal operations"""
        operations = []
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return operations
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return operations
        
        # Count protecting groups in reactants vs products
        reactant_pg_counts = self.count_protecting_groups(reactants)
        product_pg_counts = self.count_protecting_groups(products)
        
        for pg_type in self.protecting_groups:
            diff = product_pg_counts.get(pg_type, 0) - reactant_pg_counts.get(pg_type, 0)
            if diff > 0:
                operations.extend([("protection", pg_type)] * diff)
            elif diff < 0:
                operations.extend([("deprotection", pg_type)] * abs(diff))
        
        return operations
    
    def count_protecting_groups(self, mols):
        """Count occurrences of each protecting group pattern in molecules"""
        counts = {}
        
        for pg_type, pattern in self.pg_patterns.items():
            if pg_type in self.protecting_groups:
                try:
                    pg_mol = Chem.MolFromSmarts(pattern)
                    if pg_mol:
                        total_count = 0
                        for mol in mols:
                            if mol:
                                matches = mol.GetSubstructMatches(pg_mol)
                                total_count += len(matches)
                        counts[pg_type] = total_count
                except:
                    counts[pg_type] = 0
        
        return counts
    
    def evaluate_strategy(self, protection_events, deprotection_events, reactions):
        """Evaluate if the protecting group strategy meets the criteria"""
        
        # Must have at least 2 different protecting groups used
        used_pgs = set(pg for _, pg in protection_events)
        if len(used_pgs) < 2:
            return False
        
        # If sequential operations required, check that protecting groups are used at different steps
        if self.sequential_operations:
            pg_steps = {}
            for step, pg in protection_events:
                if pg not in pg_steps:
                    pg_steps[pg] = []
                pg_steps[pg].append(step)
            
            # Check that different PGs are used at different reaction steps
            step_sets = [set(steps) for steps in pg_steps.values()]
            if len(step_sets) < 2:
                return False
            
            # Ensure some PGs are used at non-overlapping steps (sequential)
            sequential_found = False
            for i in range(len(step_sets)):
                for j in range(i + 1, len(step_sets)):
                    if not step_sets[i].intersection(step_sets[j]):
                        sequential_found = True
                        break
                if sequential_found:
                    break
            
            if not sequential_found:
                return False
        
        # If multiple nitrogens required, check for diamine patterns
        if self.multiple_nitrogens:
            has_multiple_nitrogens = False
            for rxn in reactions:
                if self.contains_multiple_nitrogens(rxn):
                    has_multiple_nitrogens = True
                    break
            
            if not has_multiple_nitrogens:
                return False
        
        return True
    
    def contains_multiple_nitrogens(self, rxn):
        """Check if reaction involves molecules with multiple nitrogens"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        all_smiles = rxn_parts[0].split(".") + rxn_parts[1].split(".")
        
        for smi in all_smiles:
            smi = smi.strip()
            if smi:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        nitrogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
                        if nitrogen_count >= 2:
                            return True
                except:
                    continue
        
        return False
