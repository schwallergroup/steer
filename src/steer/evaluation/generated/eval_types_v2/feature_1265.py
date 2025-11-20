"""Generated evaluation code for: Multiple protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes based on multiple protecting group strategy.
    Checks for presence of specified protecting groups, sequential swaps,
    and phenol protection patterns.
    """
    
    def __init__(self, config):
        self.group_types = config.get("group_types", [])
        self.sequential_swaps = config.get("sequential_swaps", False)
        self.phenol_protection = config.get("phenol_protection", False)
        
        # Define protecting group SMARTS patterns
        self.pg_patterns = {
            "benzyl": "[OH1,NH1,NH2]-[CH2]-c1ccccc1",  # Benzyl protection
            "acetate": "[OH1,NH1,NH2]-C(=O)-[CH3]",     # Acetyl protection
            "MOM": "[OH1,NH1,NH2]-[CH2]-O-[CH3]"        # MOM protection
        }
        
        # Phenol pattern
        self.phenol_pattern = "c[OH1]"  # Phenolic OH
        
        # Compile patterns
        self.compiled_patterns = {}
        for name, pattern in self.pg_patterns.items():
            try:
                self.compiled_patterns[name] = Chem.MolFromSmarts(pattern)
            except:
                self.compiled_patterns[name] = None

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        pg_formations = []
        pg_removals = []
        phenol_protections = []
        
        for i, rxn in enumerate(reactions):
            # Check for protecting group formation/removal
            for pg_type in self.group_types:
                if self.detect_pg_formation(rxn, pg_type):
                    pg_formations.append((pg_type, i))
                if self.detect_pg_removal(rxn, pg_type):
                    pg_removals.append((pg_type, i))
            
            # Check for phenol protection if required
            if self.phenol_protection and self.detect_phenol_protection(rxn):
                phenol_protections.append(i)
        
        # Evaluate conditions
        condition = True
        
        # Check if all required protecting groups are used
        used_groups = set([pg for pg, _ in pg_formations + pg_removals])
        required_groups = set(self.group_types)
        if not required_groups.issubset(used_groups):
            condition = False
        
        # Check sequential swaps if required
        if self.sequential_swaps and condition:
            condition = self.check_sequential_swaps(pg_formations, pg_removals)
        
        # Check phenol protection if required
        if self.phenol_protection and condition:
            condition = len(phenol_protections) > 0
        
        return condition, len(reactions)

    def detect_pg_formation(self, rxn, pg_type):
        """Detect protecting group formation in a reaction"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            if not all(reactants + products):
                return False
            
            pattern = self.compiled_patterns.get(pg_type)
            if pattern is None:
                return False
            
            # Check if protecting group is absent in reactants but present in products
            reactant_matches = any(mol.HasSubstructMatch(pattern) for mol in reactants if mol)
            product_matches = any(mol.HasSubstructMatch(pattern) for mol in products if mol)
            
            return not reactant_matches and product_matches
            
        except:
            return False

    def detect_pg_removal(self, rxn, pg_type):
        """Detect protecting group removal in a reaction"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            if not all(reactants + products):
                return False
            
            pattern = self.compiled_patterns.get(pg_type)
            if pattern is None:
                return False
            
            # Check if protecting group is present in reactants but absent in products
            reactant_matches = any(mol.HasSubstructMatch(pattern) for mol in reactants if mol)
            product_matches = any(mol.HasSubstructMatch(pattern) for mol in products if mol)
            
            return reactant_matches and not product_matches
            
        except:
            return False

    def detect_phenol_protection(self, rxn):
        """Detect phenol protection in a reaction"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            if not all(reactants + products):
                return False
            
            phenol_pattern = Chem.MolFromSmarts(self.phenol_pattern)
            if phenol_pattern is None:
                return False
            
            # Check if phenol is protected (free phenol in reactants, protected in products)
            reactant_phenols = sum(mol.GetSubstructMatches(phenol_pattern) for mol in reactants if mol)
            product_phenols = sum(mol.GetSubstructMatches(phenol_pattern) for mol in products if mol)
            
            return len(reactant_phenols) > len(product_phenols)
            
        except:
            return False

    def check_sequential_swaps(self, formations, removals):
        """Check if protecting groups are used in sequential swap pattern"""
        if len(formations) < 2 or len(removals) < 1:
            return False
        
        # Look for pattern where one PG is removed and another is added
        for pg_type, remove_step in removals:
            # Check if there are formations of different PG types after this removal
            later_formations = [(pg, step) for pg, step in formations 
                              if step > remove_step and pg != pg_type]
            if later_formations:
                return True
        
        return False
