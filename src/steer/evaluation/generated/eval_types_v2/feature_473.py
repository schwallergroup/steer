"""Generated evaluation code for: Multiple protecting group swaps on nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwaps(MultiRxnCondBase):
    """
    Evaluates routes based on multiple protecting group swaps on nitrogen atoms.
    Checks for specific protecting group transformations (Bn, Cbz, Boc) and counts
    the number of swap operations performed.
    """
    
    def __init__(self, config):
        self.target_atom = config["parameters"]["atom"]
        self.required_swap_count = config["parameters"]["swap_count"]
        self.allowed_groups = config["parameters"]["groups"]
        
        # SMARTS patterns for protecting groups on nitrogen
        self.protecting_group_patterns = {
            "Bn": "[NH1,NH0]-[CH2]-c1ccccc1",  # Benzyl
            "Cbz": "[NH1,NH0]-C(=O)-O-[CH2]-c1ccccc1",  # Carboxybenzyl
            "Boc": "[NH1,NH0]-C(=O)-O-C(C)(C)C"  # tert-Butoxycarbonyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        swap_count = self.count_protecting_group_swaps(reactions)
        
        condition = swap_count >= self.required_swap_count
        return condition, len(reactions)
    
    def count_protecting_group_swaps(self, reactions):
        """Count the number of protecting group swap operations."""
        swap_count = 0
        
        for rxn in reactions:
            if self.is_protecting_group_swap(rxn):
                swap_count += 1
        
        return swap_count
    
    def is_protecting_group_swap(self, rxn):
        """
        Determine if a reaction represents a protecting group swap.
        A swap involves removing one protecting group and adding a different one.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Find protecting groups in products and reactants
            prod_groups = self.find_protecting_groups(prod_mol)
            react_groups = set()
            for mol in react_mols:
                react_groups.update(self.find_protecting_groups(mol))
            
            # Check if there's a net change in protecting group types
            # This indicates a swap rather than just addition/removal
            removed_groups = react_groups - prod_groups
            added_groups = prod_groups - react_groups
            
            # A swap should involve removal of one type and addition of another
            return (len(removed_groups) > 0 and len(added_groups) > 0 and 
                   removed_groups != added_groups)
            
        except:
            return False
    
    def find_protecting_groups(self, mol):
        """Find all protecting groups present in a molecule."""
        found_groups = set()
        
        for group_name, pattern in self.protecting_group_patterns.items():
            if group_name in self.allowed_groups:
                try:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        found_groups.add(group_name)
                except:
                    continue
        
        return found_groups
    
    def route_scoring(self, x):
        """
        Score based on whether the required number of swaps occurred.
        x is the fraction of route completion when condition was met.
        """
        if x < 0:
            return 0  # Condition not met
        else:
            return 10 * (1 - x)  # Earlier swaps get higher scores
