"""Generated evaluation code for: Orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on orthogonal protecting group strategy.
    Checks for the presence of specified protecting groups and whether they
    are used simultaneously and orthogonally in the route.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", [])
        self.orthogonal = config.get("orthogonal", True)
        self.simultaneous = config.get("simultaneous", True)
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[NX3][C](=O)[O][C]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "acetate": "[OH0][C](=O)[CH3]",  # acetyl ester
            "Cbz": "[NX3][C](=O)[O][CH2]c1ccccc1",  # benzyloxycarbonyl
            "TBDMS": "[OH0][Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]",  # tert-butyldimethylsilyl
            "Fmoc": "[NX3][C](=O)[O][CH2][CH]1c2ccccc2c3ccccc13",  # fluorenylmethoxycarbonyl
            "benzyl": "[OH0][CH2]c1ccccc1",  # benzyl ether
            "THP": "[OH0][CH]1[O][CH2][CH2][CH2][CH2]1"  # tetrahydropyranyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are found and in which reactions
        pg_found = {pg: [] for pg in self.protecting_groups}
        
        for i, rxn in enumerate(reactions):
            for pg in self.protecting_groups:
                if self.detect_protecting_group(rxn, pg):
                    pg_found[pg].append(i)
        
        # Check if all required protecting groups are present
        all_present = all(len(reactions_list) > 0 for reactions_list in pg_found.values())
        
        if not all_present:
            return False, len(reactions)
        
        # If simultaneous use is required, check for overlapping reaction indices
        if self.simultaneous:
            reaction_sets = [set(reactions_list) for reactions_list in pg_found.values()]
            has_overlap = any(
                len(set1.intersection(set2)) > 0 
                for i, set1 in enumerate(reaction_sets) 
                for set2 in reaction_sets[i+1:]
            )
            if not has_overlap:
                return False, len(reactions)
        
        # If orthogonal strategy is required, ensure protecting groups are chemically orthogonal
        if self.orthogonal:
            if not self.check_orthogonality():
                return False, len(reactions)
        
        return True, len(reactions)
    
    def detect_protecting_group(self, rxn, protecting_group):
        """Detect if a specific protecting group is involved in the reaction"""
        if protecting_group not in self.pg_patterns:
            return False
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[protecting_group])
        if pattern is None:
            return False
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        # Check if protecting group appears in products but not in reactants (protection)
        # or appears in reactants but not products (deprotection)
        prod_mols = [Chem.MolFromSmiles(mol) for mol in products.split(".") if mol]
        react_mols = [Chem.MolFromSmiles(mol) for mol in reactants.split(".") if mol]
        
        prod_has_pg = any(mol and mol.HasSubstructMatch(pattern) for mol in prod_mols)
        react_has_pg = any(mol and mol.HasSubstructMatch(pattern) for mol in react_mols)
        
        # Return True if there's a change in protecting group status
        return prod_has_pg != react_has_pg
    
    def check_orthogonality(self):
        """Check if the specified protecting groups are orthogonal"""
        # Define orthogonal pairs - groups that can be removed under different conditions
        orthogonal_pairs = {
            ("Boc", "acetate"), ("Boc", "benzyl"), ("Boc", "THP"),
            ("Cbz", "acetate"), ("Cbz", "TBDMS"), ("Cbz", "THP"),
            ("Fmoc", "acetate"), ("Fmoc", "TBDMS"), ("Fmoc", "benzyl"),
            ("acetate", "benzyl"), ("acetate", "TBDMS"), ("acetate", "THP"),
            ("benzyl", "TBDMS"), ("benzyl", "THP"),
            ("TBDMS", "THP")
        }
        
        # Check all pairs of protecting groups for orthogonality
        for i, pg1 in enumerate(self.protecting_groups):
            for pg2 in self.protecting_groups[i+1:]:
                pair = tuple(sorted([pg1, pg2]))
                if pair not in orthogonal_pairs:
                    return False
        
        return True
