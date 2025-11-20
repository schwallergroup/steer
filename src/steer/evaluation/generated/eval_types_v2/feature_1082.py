"""Generated evaluation code for: Multiple orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleOrthogonalProtectingGroups(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the use of multiple orthogonal protecting group strategies.
    Checks for the presence of specified protecting groups and their orthogonal deprotection.
    """
    
    def __init__(self, config):
        self.group_types = config["group_types"]
        self.require_orthogonality = config.get("orthogonality", True)
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "benzhydryl": "[CH1]([c]1[cH][cH][cH][cH][cH]1)[c]2[cH][cH][cH][cH][cH]2",  # diphenylmethyl
            "benzyl": "[CH2][c]1[cH][cH][cH][cH][cH]1",  # benzyl group
            "Cbz": "[CX3](=[OX1])[OX2][CH2][c]1[cH][cH][cH][cH][cH]1",  # benzyloxycarbonyl
            "Fmoc": "[CX3](=[OX1])[OX2][CH2][c]1[cH][cH][c]2[c]([cH]1)[c]3[cH][cH][cH][cH][c]3[CH][CH][c]2[cH][cH]",  # fluorenylmethoxycarbonyl
            "Ts": "[SX4](=[OX1])(=[OX1])([c]1[cH][cH][c]([CH3])[cH][cH]1)",  # tosyl
            "Ms": "[SX4](=[OX1])(=[OX1])([CH3])",  # mesyl
            "TBS": "[SiX4]([CX4]([CH3])([CH3])[CH3])([CH3])([CH3])"  # tert-butyldimethylsilyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are used and removed
        groups_installed = set()
        groups_removed = set()
        
        for rxn in reactions:
            # Check for protection reactions (installing groups)
            for group_name in self.group_types:
                if self.detect_protection(rxn, group_name):
                    groups_installed.add(group_name)
                elif self.detect_deprotection(rxn, group_name):
                    groups_removed.add(group_name)
        
        # Check if required groups are present
        required_groups_found = all(group in groups_installed for group in self.group_types)
        
        # Check orthogonality if required
        orthogonal_use = True
        if self.require_orthogonality and len(self.group_types) > 1:
            # Ensure groups are removed selectively (not all at once)
            orthogonal_use = self.check_orthogonal_deprotection(reactions)
        
        condition_met = required_groups_found and orthogonal_use
        return condition_met, len(reactions)
    
    def detect_protection(self, rxn, group_name):
        """Detect if a protecting group is being installed"""
        if group_name not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[group_name])
        if pattern is None:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        # Check if group appears in product but not in reactants
        product = Chem.MolFromSmiles(rxn_parts[1])
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".") if r]
        
        if product is None:
            return False
            
        product_has_group = product.HasSubstructMatch(pattern)
        reactant_has_group = any(r and r.HasSubstructMatch(pattern) for r in reactants if r)
        
        # Protection: group in product but not in reactants
        return product_has_group and not reactant_has_group
    
    def detect_deprotection(self, rxn, group_name):
        """Detect if a protecting group is being removed"""
        if group_name not in self.pg_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.pg_patterns[group_name])
        if pattern is None:
            return False
            
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        # Check if group appears in reactants but not in product
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".") if r]
        product = Chem.MolFromSmiles(rxn_parts[1])
        
        if product is None:
            return False
            
        reactant_has_group = any(r and r.HasSubstructMatch(pattern) for r in reactants if r)
        product_has_group = product.HasSubstructMatch(pattern)
        
        # Deprotection: group in reactants but not in product
        return reactant_has_group and not product_has_group
    
    def check_orthogonal_deprotection(self, reactions):
        """Check if protecting groups are removed orthogonally (separately)"""
        deprotection_steps = []
        
        for rxn in reactions:
            groups_removed_in_step = []
            for group_name in self.group_types:
                if self.detect_deprotection(rxn, group_name):
                    groups_removed_in_step.append(group_name)
            
            if groups_removed_in_step:
                deprotection_steps.append(groups_removed_in_step)
        
        # Orthogonality means no step removes more than one type of protecting group
        # unless they are the same type
        for step in deprotection_steps:
            unique_groups = set(step)
            if len(unique_groups) > 1:
                return False
        
        return True
