"""Generated evaluation code for: Protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for protecting group swap strategies.
    Checks if a route involves deprotection of an initial protecting group
    followed by reprotection with a different protecting group.
    """
    
    def __init__(self, config):
        self.initial_group = config["parameters"]["initial_group"]
        self.final_group = config["parameters"]["final_group"]
        self.swap_location = config["parameters"]["swap_location"]
        
        # Define SMARTS patterns for common protecting groups
        self.protecting_group_patterns = {
            "Boc": "[CH3:1][CH3:2][CH3:3][C:4](=[O:5])[O:6][C:7](=[O:8])[N:9]",
            "trifluoroacetyl": "[F:1][C:2]([F:3])([F:4])[C:5](=[O:6])[N:7]",
            "Cbz": "[c:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH2:7][O:8][C:9](=[O:10])[N:11]",
            "Fmoc": "[c:1]1[cH:2][cH:3][c:4]2[c:5]([cH:6]1)[cH:7][c:8]3[cH:9][cH:10][cH:11][cH:12][c:13]3[c:14]2[CH:15]([CH2:16][O:17][C:18](=[O:19])[N:20])",
            "acetyl": "[CH3:1][C:2](=[O:3])[N:4]",
            "Ts": "[CH3:1][c:2]1[cH:3][cH:4][c:5]([cH:6][cH:7]1)[S:8](=[O:9])(=[O:10])[N:11]"
        }
    
    def condition_depth(self, d):
        """
        Check if the route contains the protecting group swap strategy.
        Returns (condition_met, total_reactions).
        """
        reactions = self.get_rxns(d)
        
        # Find deprotection and reprotection events
        deprotection_step = -1
        reprotection_step = -1
        
        for i, rxn in enumerate(reactions):
            if self.is_deprotection(rxn, self.initial_group):
                deprotection_step = i
            elif self.is_protection(rxn, self.final_group) and deprotection_step >= 0:
                reprotection_step = i
                break
        
        # Check if both events occurred and meet location criteria
        condition_met = self.evaluate_swap_strategy(deprotection_step, reprotection_step, len(reactions))
        
        return condition_met, len(reactions)
    
    def is_deprotection(self, rxn_smiles, protecting_group):
        """Check if reaction involves removal of specified protecting group."""
        if protecting_group not in self.protecting_group_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.protecting_group_patterns[protecting_group])
        if pattern is None:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactant = Chem.MolFromSmiles(parts[0])
        products = [Chem.MolFromSmiles(p) for p in parts[1].split(".")]
        
        if reactant is None or any(p is None for p in products):
            return False
            
        # Check if protecting group is in reactant but not in main product
        has_pg_in_reactant = reactant.HasSubstructMatch(pattern)
        has_pg_in_products = any(p.HasSubstructMatch(pattern) for p in products)
        
        return has_pg_in_reactant and not has_pg_in_products
    
    def is_protection(self, rxn_smiles, protecting_group):
        """Check if reaction involves addition of specified protecting group."""
        if protecting_group not in self.protecting_group_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.protecting_group_patterns[protecting_group])
        if pattern is None:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r) for r in parts[0].split(".")]
        product = Chem.MolFromSmiles(parts[1])
        
        if any(r is None for r in reactants) or product is None:
            return False
            
        # Check if protecting group is not in reactants but is in product
        has_pg_in_reactants = any(r.HasSubstructMatch(pattern) for r in reactants)
        has_pg_in_product = product.HasSubstructMatch(pattern)
        
        return not has_pg_in_reactants and has_pg_in_product
    
    def evaluate_swap_strategy(self, deprotection_step, reprotection_step, total_reactions):
        """Evaluate if the swap strategy meets the specified location criteria."""
        if deprotection_step == -1 or reprotection_step == -1:
            return False
            
        if deprotection_step >= reprotection_step:
            return False  # Deprotection should come before reprotection
            
        if self.swap_location == "early_route":
            # Both steps should occur in first third of route
            return deprotection_step < total_reactions / 3 and reprotection_step < total_reactions / 3
        elif self.swap_location == "mid_route":
            # Steps should occur in middle third of route
            return (total_reactions / 3 <= deprotection_step < 2 * total_reactions / 3 and
                    total_reactions / 3 <= reprotection_step < 2 * total_reactions / 3)
        elif self.swap_location == "late_route":
            # Steps should occur in final third of route
            return (deprotection_step >= 2 * total_reactions / 3 and
                    reprotection_step >= 2 * total_reactions / 3)
        else:
            # Any location is acceptable
            return True
