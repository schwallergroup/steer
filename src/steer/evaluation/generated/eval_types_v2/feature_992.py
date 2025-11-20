"""Generated evaluation code for: Multiple protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on the use of multiple different protecting groups
    that are added and later removed (cycling). Checks for the presence of specified
    protecting group types and counts how many different types are cycled.
    """
    
    def __init__(self, config):
        self.pg_types = config.get("pg_types", ["PMB", "Benzyl", "Boc"])
        self.cycling_count = config.get("cycling_count", 3)
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "PMB": "[CH2]c1ccc(OC)cc1",  # para-methoxybenzyl
            "Benzyl": "[CH2]c1ccccc1",   # benzyl
            "Boc": "C(=O)OC(C)(C)C"      # tert-butoxycarbonyl
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group additions and removals
        pg_added = set()
        pg_removed = set()
        
        for rxn in reactions:
            added, removed = self.analyze_pg_changes(rxn)
            pg_added.update(added)
            pg_removed.update(removed)
        
        # Count how many different PG types were both added and removed (cycled)
        cycled_pgs = pg_added.intersection(pg_removed)
        cycled_count = len(cycled_pgs)
        
        condition = cycled_count >= self.cycling_count
        return condition, len(reactions)

    def analyze_pg_changes(self, rxn):
        """
        Analyze a reaction to determine which protecting groups were added or removed.
        Returns tuple of (added_pgs, removed_pgs) as sets.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return set(), set()
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Filter out None molecules
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        if not reactants or not products:
            return set(), set()
        
        # Count PG occurrences in reactants and products
        reactant_pg_counts = self.count_protecting_groups(reactants)
        product_pg_counts = self.count_protecting_groups(products)
        
        added_pgs = set()
        removed_pgs = set()
        
        for pg_type in self.pg_types:
            reactant_count = reactant_pg_counts.get(pg_type, 0)
            product_count = product_pg_counts.get(pg_type, 0)
            
            if product_count > reactant_count:
                added_pgs.add(pg_type)
            elif reactant_count > product_count:
                removed_pgs.add(pg_type)
        
        return added_pgs, removed_pgs

    def count_protecting_groups(self, molecules):
        """
        Count occurrences of each protecting group type in a list of molecules.
        Returns dictionary mapping PG type to total count.
        """
        pg_counts = {}
        
        for pg_type in self.pg_types:
            if pg_type in self.pg_patterns:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
                if pattern is not None:
                    total_count = 0
                    for mol in molecules:
                        if mol is not None:
                            matches = mol.GetSubstructMatches(pattern)
                            total_count += len(matches)
                    pg_counts[pg_type] = total_count
        
        return pg_counts
