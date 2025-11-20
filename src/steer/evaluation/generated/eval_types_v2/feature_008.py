"""Generated evaluation code for: Complex protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes for complex protecting group cycling strategies.
    Checks for the presence of specified protecting groups, cycle count,
    and final product transformation.
    """
    
    def __init__(self, config):
        self.pg_types = config["pg_types"]
        self.cycle_count = config["cycle_count"]
        self.final_product = config["final_product"]
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "tert_butyl_ester": "[C](=O)OC(C)(C)C",
            "benzyl_ester": "[C](=O)OCc1ccccc1",
            "methyl_ester": "[C](=O)OC"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group operations
        pg_installations = {pg: 0 for pg in self.pg_types}
        pg_removals = {pg: 0 for pg in self.pg_types}
        final_product_formed = False
        
        for rxn in reactions:
            # Check for protecting group installations and removals
            for pg_type in self.pg_types:
                if self.detect_pg_installation(rxn, pg_type):
                    pg_installations[pg_type] += 1
                elif self.detect_pg_removal(rxn, pg_type):
                    pg_removals[pg_type] += 1
            
            # Check for final product formation
            if self.detect_final_product_formation(rxn):
                final_product_formed = True
        
        # Evaluate strategy conditions
        strategy_met = self.evaluate_strategy(pg_installations, pg_removals, final_product_formed)
        
        return strategy_met, len(reactions)
    
    def detect_pg_installation(self, rxn, pg_type):
        """Detect installation of a protecting group"""
        if pg_type not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
        if pattern is None:
            return False
        
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Count PG in reactants vs products
            reactant_matches = sum(1 for mol in reactants if mol and mol.HasSubstructMatch(pattern))
            product_matches = sum(1 for mol in products if mol and mol.HasSubstructMatch(pattern))
            
            return product_matches > reactant_matches
            
        except:
            return False
    
    def detect_pg_removal(self, rxn, pg_type):
        """Detect removal of a protecting group"""
        if pg_type not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
        if pattern is None:
            return False
        
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Count PG in reactants vs products
            reactant_matches = sum(1 for mol in reactants if mol and mol.HasSubstructMatch(pattern))
            product_matches = sum(1 for mol in products if mol and mol.HasSubstructMatch(pattern))
            
            return reactant_matches > product_matches
            
        except:
            return False
    
    def detect_final_product_formation(self, rxn):
        """Detect formation of the final product type"""
        if self.final_product not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[self.final_product])
        if pattern is None:
            return False
        
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Check if final product is formed
            reactant_matches = sum(1 for mol in reactants if mol and mol.HasSubstructMatch(pattern))
            product_matches = sum(1 for mol in products if mol and mol.HasSubstructMatch(pattern))
            
            return product_matches > reactant_matches
            
        except:
            return False
    
    def evaluate_strategy(self, pg_installations, pg_removals, final_product_formed):
        """Evaluate if the protecting group strategy is met"""
        # Check if all required PG types were used
        all_pgs_used = all(pg_installations[pg] > 0 for pg in self.pg_types)
        
        # Check if cycling occurred (installation followed by removal)
        cycling_count = sum(min(pg_installations[pg], pg_removals[pg]) for pg in self.pg_types)
        sufficient_cycling = cycling_count >= self.cycle_count
        
        # Check if final product was formed
        return all_pgs_used and sufficient_cycling and final_product_formed
    
    def route_scoring(self, x):
        """Convert strategy evaluation to score"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 1 - x  # Earlier implementation is better
