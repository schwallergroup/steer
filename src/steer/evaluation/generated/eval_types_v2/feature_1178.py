"""Generated evaluation code for: Sequential dual protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes for sequential dual protecting group strategy.
    Checks if the route uses multiple orthogonal protecting groups (DMB, Cbz, trityl) 
    in sequence to differentiate amine sites.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["protecting_groups"]
        self.strategy_type = config["strategy_type"]
        self.amine_sites = config["amine_sites"]
        
        # SMARTS patterns for protecting groups
        self.pg_patterns = {
            "DMB": "[NH1,NH2]-C(=O)-O-c1ccccc1C",  # Dimethoxybenzyl carbamate
            "Cbz": "[NH1,NH2]-C(=O)-O-Cc1ccccc1",   # Benzyloxycarbonyl
            "trityl": "[NH1,NH2]-C(c1ccccc1)(c2ccccc2)c3ccccc3"  # Trityl
        }
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group installations and removals
        pg_installations = []
        pg_removals = []
        
        for i, rxn in enumerate(reactions):
            for pg_name in self.protecting_groups:
                if self.detect_pg_installation(rxn, pg_name):
                    pg_installations.append((pg_name, i))
                elif self.detect_pg_removal(rxn, pg_name):
                    pg_removals.append((pg_name, i))
        
        # Check if strategy requirements are met
        condition = self.evaluate_strategy(pg_installations, pg_removals)
        
        return condition, len(reactions)
    
    def detect_pg_installation(self, rxn, pg_name):
        """Detect protecting group installation reaction"""
        if pg_name not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
        if pattern is None:
            return False
            
        # Check if product has the protecting group but reactants don't
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Count protecting groups in reactants vs products
        reactant_count = sum(self.count_pg_in_mol(r, pattern) for r in reactants)
        product_count = sum(self.count_pg_in_mol(p, pattern) for p in products)
        
        return product_count > reactant_count
    
    def detect_pg_removal(self, rxn, pg_name):
        """Detect protecting group removal reaction"""
        if pg_name not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
        if pattern is None:
            return False
            
        # Check if reactant has the protecting group but product doesn't
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Count protecting groups in reactants vs products
        reactant_count = sum(self.count_pg_in_mol(r, pattern) for r in reactants)
        product_count = sum(self.count_pg_in_mol(p, pattern) for p in products)
        
        return reactant_count > product_count
    
    def count_pg_in_mol(self, smiles, pattern):
        """Count occurrences of protecting group pattern in molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            return len(mol.GetSubstructMatches(pattern))
        except:
            return 0
    
    def evaluate_strategy(self, installations, removals):
        """Evaluate if the protecting group strategy meets requirements"""
        # Check if we have installations of the required protecting groups
        installed_groups = set(pg for pg, _ in installations)
        required_groups = set(self.protecting_groups)
        
        # Must have at least the minimum required protecting groups
        if len(installed_groups.intersection(required_groups)) < min(2, len(required_groups)):
            return False
        
        # For sequential orthogonal strategy, check temporal ordering
        if self.strategy_type == "sequential_orthogonal":
            return self.check_sequential_orthogonal(installations, removals)
        
        return True
    
    def check_sequential_orthogonal(self, installations, removals):
        """Check if protecting groups are used in orthogonal sequential manner"""
        # Group by protecting group type
        pg_timeline = {}
        
        for pg, step in installations:
            if pg not in pg_timeline:
                pg_timeline[pg] = {"install": [], "remove": []}
            pg_timeline[pg]["install"].append(step)
        
        for pg, step in removals:
            if pg not in pg_timeline:
                pg_timeline[pg] = {"install": [], "remove": []}
            pg_timeline[pg]["remove"].append(step)
        
        # Check that different protecting groups are used at different stages
        used_groups = []
        for pg in self.protecting_groups:
            if pg in pg_timeline and pg_timeline[pg]["install"]:
                used_groups.append(pg)
        
        # Need at least 2 different protecting groups for dual strategy
        if len(used_groups) < 2:
            return False
        
        # Check that they don't significantly overlap in timing
        install_times = []
        for pg in used_groups:
            if pg in pg_timeline:
                install_times.extend(pg_timeline[pg]["install"])
        
        # Sequential means installations should be spread out
        if len(set(install_times)) >= 2:
            return True
        
        return False
