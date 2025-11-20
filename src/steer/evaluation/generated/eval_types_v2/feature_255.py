"""Generated evaluation code for: Multiple protecting group strategy with trityl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes that employ multiple orthogonal protecting groups with a specific final deprotection step.
    Checks for the presence of specified protecting groups and validates the final deprotection strategy.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["protecting_groups"]
        self.required_count = config["count"]
        self.final_deprotection = config["final_deprotection"]
        
        # Define SMARTS patterns for protecting group detection
        self.pg_patterns = {
            "trityl": "[CH]([c]1[cH][cH][cH][cH][cH]1)([c]2[cH][cH][cH][cH][cH]2)[c]3[cH][cH][cH][cH][cH]3",
            "benzyl": "[CH2][c]1[cH][cH][cH][cH][cH]1",
            "MOM": "[CH2][O][CH3]",
            "Boc": "[C](=[O])[O][C]([CH3])([CH3])[CH3]",
            "TBDMS": "[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]",
            "acetyl": "[C](=[O])[CH3]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are used
        used_pgs = set()
        final_deprotection_found = False
        
        for i, rxn in enumerate(reactions):
            # Check for protecting group installation or removal
            for pg_name in self.protecting_groups:
                if self.detect_protecting_group_reaction(rxn, pg_name):
                    used_pgs.add(pg_name)
                    
                    # Check if this is the final deprotection step (last reaction involving the target PG)
                    if (pg_name == self.final_deprotection and 
                        self.is_deprotection_reaction(rxn, pg_name) and
                        i == len(reactions) - 1):
                        final_deprotection_found = True
        
        # Condition is met if we have the required number of different protecting groups
        # and the final deprotection is the specified one
        condition = (len(used_pgs) >= self.required_count and 
                    self.final_deprotection in used_pgs and
                    final_deprotection_found)
        
        return condition, len(reactions)
    
    def detect_protecting_group_reaction(self, rxn, pg_name):
        """Detect if a protecting group is involved in the reaction (installation or removal)"""
        if pg_name not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
        if pattern is None:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if protecting group appears or disappears
        reactant_has_pg = any(self.mol_has_pattern(r, pattern) for r in reactants)
        product_has_pg = any(self.mol_has_pattern(p, pattern) for p in products)
        
        return reactant_has_pg != product_has_pg  # XOR - appears or disappears
    
    def is_deprotection_reaction(self, rxn, pg_name):
        """Check if this is a deprotection reaction (protecting group is removed)"""
        if pg_name not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
        if pattern is None:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Deprotection: protecting group present in reactants but not in products
        reactant_has_pg = any(self.mol_has_pattern(r, pattern) for r in reactants)
        product_has_pg = any(self.mol_has_pattern(p, pattern) for p in products)
        
        return reactant_has_pg and not product_has_pg
    
    def mol_has_pattern(self, smiles, pattern):
        """Check if molecule contains the specified pattern"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return mol.HasSubstructMatch(pattern)
        except:
            return False
