"""Generated evaluation code for: Sequential amine protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialAmineProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for sequential amine protecting group cycling patterns.
    Checks for protection/deprotection sequences with specified protecting groups.
    """
    
    def __init__(self, config):
        self.functional_group = config["functional_group"]
        self.protection_count = config["protection_count"]
        self.deprotection_count = config["deprotection_count"]
        self.sequential_cycling = config["sequential_cycling"]
        self.protecting_groups = config["protecting_groups"]
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "ethyl_carbamate": "[NH1,NH0]-C(=O)O[CH2][CH3]",  # N-COOEt
            "formyl": "[NH1,NH0]-C(=O)[H]"  # N-CHO
        }
        
        # Pattern for free amine
        self.free_amine_pattern = "[NH2,NH1][C,c]"

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < (self.protection_count + self.deprotection_count):
            return False, len(reactions)
        
        # Track protection/deprotection events
        pg_events = []
        for i, rxn in enumerate(reactions):
            event = self.classify_pg_event(rxn)
            if event:
                pg_events.append((i, event))
        
        # Check if we have the required cycling pattern
        condition = self.has_sequential_cycling(pg_events)
        return condition, len(reactions)
    
    def classify_pg_event(self, rxn):
        """Classify reaction as protection, deprotection, or neither"""
        reactants_smiles = rxn.split(">>")[0]
        products_smiles = rxn.split(">>")[1]
        
        reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
        products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
        
        if not all(reactants) or not all(products):
            return None
        
        # Count protecting groups and free amines in reactants vs products
        reactant_pg_counts = {pg: sum(self.count_pattern(mol, pattern) 
                                    for mol in reactants) 
                            for pg, pattern in self.pg_patterns.items()}
        product_pg_counts = {pg: sum(self.count_pattern(mol, pattern) 
                                   for mol in products) 
                           for pg, pattern in self.pg_patterns.items()}
        
        reactant_free_amines = sum(self.count_pattern(mol, self.free_amine_pattern) 
                                 for mol in reactants)
        product_free_amines = sum(self.count_pattern(mol, self.free_amine_pattern) 
                                for mol in products)
        
        # Determine if this is protection or deprotection
        for pg in self.protecting_groups:
            if product_pg_counts[pg] > reactant_pg_counts[pg]:
                return f"protect_{pg}"
            elif reactant_pg_counts[pg] > product_pg_counts[pg]:
                return f"deprotect_{pg}"
        
        return None
    
    def count_pattern(self, mol, pattern):
        """Count occurrences of SMARTS pattern in molecule"""
        if mol is None:
            return 0
        patt = Chem.MolFromSmarts(pattern)
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt))
    
    def has_sequential_cycling(self, pg_events):
        """Check if protection/deprotection events match the required cycling pattern"""
        if len(pg_events) < (self.protection_count + self.deprotection_count):
            return False
        
        # Look for the specific pattern: protect with ethyl_carbamate, deprotect, protect with formyl
        protection_events = 0
        deprotection_events = 0
        last_protection_type = None
        
        for depth, event in pg_events:
            if event.startswith("protect_"):
                protection_events += 1
                pg_type = event.replace("protect_", "")
                
                # Check sequential order for cycling
                if self.sequential_cycling:
                    if protection_events == 1 and pg_type != "ethyl_carbamate":
                        return False
                    elif protection_events == 2 and pg_type != "formyl":
                        return False
                
                last_protection_type = pg_type
                
            elif event.startswith("deprotect_"):
                deprotection_events += 1
                pg_type = event.replace("deprotect_", "")
                
                # Deprotection should match the last protection type
                if last_protection_type and pg_type != last_protection_type:
                    return False
        
        # Check if we have the right counts
        return (protection_events >= self.protection_count and 
                deprotection_events >= self.deprotection_count)
