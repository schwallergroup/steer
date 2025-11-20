"""Generated evaluation code for: Protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for protecting group swap strategies.
    Checks if a first protecting group is removed and then a second protecting group
    is added in sequential steps.
    """
    
    def __init__(self, config):
        self.first_group = config["first_group"]
        self.second_group = config["second_group"]
        self.swap_type = config.get("swap_type", "sequential")
        
        # Define protecting group SMARTS patterns
        self.protecting_patterns = {
            "Cbz": "[#6]-[#6](=[#8])-[#8]-[#6]-c1ccccc1",  # Benzyl carbamate
            "TFA": "[#6]-[#6](=[#8])-[#6]([#9])([#9])[#9]",  # Trifluoroacetyl
            "Boc": "[#6]-[#6](=[#8])-[#8]-[#6]([#6])([#6])[#6]",  # tert-Butoxycarbonyl
            "Fmoc": "[#6]-[#6](=[#8])-[#8]-[#6]-c1c2ccccc2c3ccccc13",  # Fluorenylmethoxycarbonyl
            "Ac": "[#6]-[#6](=[#8])-[#6]",  # Acetyl
            "Ts": "[#6]-[#16](=[#8])(=[#8])-c1ccc([#6])cc1"  # Tosyl
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the protecting group swap strategy is present in the route.
        Returns (condition_met, total_reactions)
        """
        reactions = self.get_rxns(d)
        
        if self.swap_type == "sequential":
            condition = self._check_sequential_swap(reactions)
        else:
            condition = False
            
        return condition, len(reactions)

    def _check_sequential_swap(self, reactions) -> bool:
        """
        Check for sequential protecting group swap: removal followed by addition.
        """
        deprotection_found = False
        protection_found = False
        deprotection_index = -1
        
        for i, rxn in enumerate(reactions):
            # Check for deprotection of first group
            if self._is_deprotection(rxn, self.first_group):
                deprotection_found = True
                deprotection_index = i
            
            # Check for protection with second group (must come after deprotection)
            elif deprotection_found and i > deprotection_index and self._is_protection(rxn, self.second_group):
                protection_found = True
                break
                
        return deprotection_found and protection_found

    def _is_deprotection(self, rxn, protecting_group) -> bool:
        """
        Check if a reaction involves removal of a specific protecting group.
        """
        if protecting_group not in self.protecting_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.protecting_patterns[protecting_group])
        if pattern is None:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if protecting group is present in reactants but absent in products
        reactant_has_pg = False
        product_has_pg = False
        
        for r_smi in reactants.split("."):
            mol = Chem.MolFromSmiles(r_smi)
            if mol and mol.HasSubstructMatch(pattern):
                reactant_has_pg = True
                break
                
        for p_smi in products.split("."):
            mol = Chem.MolFromSmiles(p_smi)
            if mol and mol.HasSubstructMatch(pattern):
                product_has_pg = True
                break
                
        return reactant_has_pg and not product_has_pg

    def _is_protection(self, rxn, protecting_group) -> bool:
        """
        Check if a reaction involves addition of a specific protecting group.
        """
        if protecting_group not in self.protecting_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.protecting_patterns[protecting_group])
        if pattern is None:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if protecting group is absent in reactants but present in products
        reactant_has_pg = False
        product_has_pg = False
        
        for r_smi in reactants.split("."):
            mol = Chem.MolFromSmiles(r_smi)
            if mol and mol.HasSubstructMatch(pattern):
                reactant_has_pg = True
                break
                
        for p_smi in products.split("."):
            mol = Chem.MolFromSmiles(p_smi)
            if mol and mol.HasSubstructMatch(pattern):
                product_has_pg = True
                break
                
        return not reactant_has_pg and product_has_pg
