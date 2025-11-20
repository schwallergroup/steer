"""Generated evaluation code for: Orthogonal protecting group strategy using PMB, TBDMS, Bz"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the use of orthogonal protecting group strategies.
    Checks for the presence of PMB, TBDMS, and Bz protecting groups that can be
    selectively removed under different conditions.
    """
    
    def __init__(self, config):
        self.required_groups = config["parameters"]["protecting_groups"]
        self.strategy = config["parameters"]["strategy"]
        self.selectivity = config["parameters"]["selectivity"]
        
        # SMARTS patterns for each protecting group
        self.pg_patterns = {
            "PMB": "[CH2]c1ccc(OC)cc1",  # para-methoxybenzyl
            "TBDMS": "[Si](C)(C)C(C)(C)C",  # tert-butyldimethylsilyl
            "Bz": "C(=O)c1ccccc1"  # benzoyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are used
        groups_found = set()
        
        for rxn in reactions:
            for pg_name in self.required_groups:
                if self.detect_protecting_group_use(rxn, pg_name):
                    groups_found.add(pg_name)
        
        # Check if strategy requirements are met
        condition_met = self.evaluate_strategy(groups_found)
        
        return condition_met, len(reactions)
    
    def detect_protecting_group_use(self, rxn, pg_name):
        """
        Detect if a specific protecting group is introduced or removed in the reaction.
        """
        if pg_name not in self.pg_patterns:
            return False
            
        pattern = self.pg_patterns[pg_name]
        mol_pattern = Chem.MolFromSmarts(pattern)
        
        if mol_pattern is None:
            return False
        
        # Check both protection and deprotection reactions
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Parse reactants and products
        reactant_mols = []
        for smi in reactants.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                reactant_mols.append(mol)
        
        product_mols = []
        for smi in products.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                product_mols.append(mol)
        
        # Count occurrences in reactants and products
        reactant_matches = sum(len(mol.GetSubstructMatches(mol_pattern)) 
                              for mol in reactant_mols)
        product_matches = sum(len(mol.GetSubstructMatches(mol_pattern)) 
                             for mol in product_mols)
        
        # Protection: PG appears in products but not reactants (net increase)
        # Deprotection: PG disappears from reactants to products (net decrease)
        return reactant_matches != product_matches
    
    def evaluate_strategy(self, groups_found):
        """
        Evaluate if the protecting group strategy meets the requirements.
        """
        if self.strategy == "orthogonal":
            # For orthogonal strategy, we need all specified protecting groups
            required_set = set(self.required_groups)
            return required_set.issubset(groups_found)
        
        elif self.strategy == "selective":
            # For selective strategy, we need at least 2 different groups
            return len(groups_found) >= 2
        
        else:
            # Default: at least one of the specified groups should be present
            return len(groups_found) > 0
    
    def route_scoring(self, depth_fraction):
        """
        Convert depth fraction to score. Earlier use of orthogonal protection
        strategy is generally better for synthetic planning.
        """
        if depth_fraction < 0:
            return 0  # Strategy not found
        
        # Score from 0-10, with earlier implementation scoring higher
        return 10 * (1 - depth_fraction)
