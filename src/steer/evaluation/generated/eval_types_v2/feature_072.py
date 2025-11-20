"""Generated evaluation code for: Multiple protecting group strategy employed"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route employs multiple distinct protecting group types.
    Checks for the presence of specified protecting groups (Boc, SEM, benzyl) and ensures
    the minimum number of different types are used throughout the route.
    """
    
    def __init__(self, config):
        self.min_pg_types = config.get("min_pg_types", 3)
        self.target_pg_types = config.get("pg_types", ["Boc", "SEM", "benzyl"])
        
        # Define SMARTS patterns for each protecting group type
        self.pg_patterns = {
            "Boc": "[#6](=O)O[#6]([#6])([#6])[#6]",  # tert-butoxycarbonyl
            "SEM": "[#6][Si]([#6])([#6])O[#6][#6]O[#6]",  # 2-(trimethylsilyl)ethoxymethyl
            "benzyl": "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1[#6]O"  # benzyl ether/ester
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are found
        found_pg_types = set()
        
        for rxn in reactions:
            for pg_type in self.target_pg_types:
                if self.detect_protecting_group(rxn, pg_type):
                    found_pg_types.add(pg_type)
        
        # Check if we have the minimum number of different PG types
        condition_met = len(found_pg_types) >= self.min_pg_types
        
        # Return condition result and number of distinct PG types found
        return condition_met, len(found_pg_types)
    
    def detect_protecting_group(self, rxn, pg_type):
        """
        Detect if a specific protecting group type is involved in a reaction.
        Checks both protection (formation) and deprotection (removal) reactions.
        """
        if pg_type not in self.pg_patterns:
            return False
            
        pattern = self.pg_patterns[pg_type]
        mol_pattern = Chem.MolFromSmarts(pattern)
        
        if mol_pattern is None:
            return False
        
        # Parse reaction SMILES
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Check reactants and products for protecting group pattern
        reactant_has_pg = False
        product_has_pg = False
        
        # Check reactants
        for reactant_smi in reactants_smiles.split("."):
            reactant_mol = Chem.MolFromSmiles(reactant_smi)
            if reactant_mol and reactant_mol.HasSubstructMatch(mol_pattern):
                reactant_has_pg = True
                break
        
        # Check products
        for product_smi in products_smiles.split("."):
            product_mol = Chem.MolFromSmiles(product_smi)
            if product_mol and product_mol.HasSubstructMatch(mol_pattern):
                product_has_pg = True
                break
        
        # Protecting group is involved if it appears in reactants OR products
        # (either protection or deprotection reaction)
        return reactant_has_pg or product_has_pg
    
    def route_scoring(self, x) -> float:
        """
        Score based on how many distinct protecting group types are used.
        Higher scores for meeting or exceeding the minimum requirement.
        """
        if x >= self.min_pg_types:
            # Reward for meeting minimum + bonus for additional diversity
            return min(10.0, 7.0 + (x - self.min_pg_types) * 1.5)
        else:
            # Partial credit based on how many PG types were found
            return (x / self.min_pg_types) * 5.0
