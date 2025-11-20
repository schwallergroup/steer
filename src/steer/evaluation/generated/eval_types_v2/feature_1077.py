"""Generated evaluation code for: Multiple protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on the use of multiple protecting group strategies.
    Checks for the presence of specified protecting groups and their orthogonal usage.
    """
    
    def __init__(self, config):
        self.pg_types = config.get("pg_types", [])
        self.orthogonality = config.get("orthogonality", "mixed")  # "mixed", "orthogonal", "same"
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "TBS": "[Si](C)(C)C(C)(C)C",  # tert-butyldimethylsilyl
            "PMB": "COc1ccc(CO)cc1",  # para-methoxybenzyl
            "acetate": "CC(=O)O",  # acetyl
            "pivaloate": "CC(C)(C)C(=O)O"  # pivaloyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are found
        found_pgs = set()
        pg_reaction_count = 0
        
        for rxn in reactions:
            pg_found_in_rxn = self.detect_protecting_groups(rxn)
            if pg_found_in_rxn:
                found_pgs.update(pg_found_in_rxn)
                pg_reaction_count += 1
        
        # Check if strategy requirements are met
        condition_met = self.evaluate_strategy(found_pgs, pg_reaction_count)
        
        return condition_met, len(reactions)
    
    def detect_protecting_groups(self, rxn):
        """Detect protecting group installation/removal in a reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return set()
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            reactants = [Chem.MolFromSmiles(s.strip()) for s in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(s.strip()) for s in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
        except:
            return set()
        
        found_pgs = set()
        
        # Check for protecting group patterns in reactants and products
        for pg_type in self.pg_types:
            if pg_type in self.pg_patterns:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
                if pattern is not None:
                    # Check if PG appears or disappears (protection/deprotection)
                    reactant_has_pg = any(mol.HasSubstructMatch(pattern) for mol in reactants)
                    product_has_pg = any(mol.HasSubstructMatch(pattern) for mol in products)
                    
                    if reactant_has_pg != product_has_pg:  # PG added or removed
                        found_pgs.add(pg_type)
        
        return found_pgs
    
    def evaluate_strategy(self, found_pgs, pg_reaction_count):
        """Evaluate if the protecting group strategy meets requirements"""
        if len(found_pgs) < 2:
            return False  # Need multiple PGs for a strategy
            
        if self.orthogonality == "mixed":
            # Require at least 2 different PG types from the specified list
            target_pgs = set(self.pg_types) & found_pgs
            return len(target_pgs) >= 2
            
        elif self.orthogonality == "orthogonal":
            # Require orthogonal PGs (different removal conditions)
            orthogonal_pairs = [
                ("TBS", "acetate"),
                ("TBS", "PMB"), 
                ("PMB", "pivaloate"),
                ("acetate", "pivaloate")
            ]
            for pg1, pg2 in orthogonal_pairs:
                if pg1 in found_pgs and pg2 in found_pgs:
                    return True
            return False
            
        elif self.orthogonality == "same":
            # All PGs should be of the same type (non-orthogonal strategy)
            return len(found_pgs) == 1 and pg_reaction_count >= 2
            
        return False
    
    def route_scoring(self, x):
        """Convert condition result to score (0-1 range)"""
        if x < 0:
            return 0  # Strategy not found
        else:
            return 1 - x  # Earlier implementation of strategy is better
