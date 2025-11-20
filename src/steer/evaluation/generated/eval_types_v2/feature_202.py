"""Generated evaluation code for: Sequential protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential protecting group cycling strategy.
    Checks if the route follows a specified sequence of protecting/deprotecting reactions
    for a given number of cycles.
    """
    
    def __init__(self, config):
        self.protection_sequence = config["protection_sequence"]
        self.cycle_count = config["cycle_count"]
        
        # Define SMARTS patterns for common protecting groups
        self.protecting_group_patterns = {
            "Boc": "[N:1][C:2](=O)[O:3][C:4]([CH3:5])([CH3:6])[CH3:7]",  # tert-butoxycarbonyl
            "Cbz": "[N:1][C:2](=O)[O:3][CH2:4][c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1",  # benzyloxycarbonyl
            "free_amine": "[NH2:1]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        sequence_matches = self.detect_protecting_group_sequence(reactions)
        
        # Check if we have the required number of complete cycles
        complete_cycles = sequence_matches // len(self.protection_sequence)
        condition = complete_cycles >= self.cycle_count
        
        return condition, len(reactions)
    
    def detect_protecting_group_sequence(self, reactions):
        """
        Detect how many times the complete protection sequence occurs in order.
        Returns the number of sequence steps matched.
        """
        sequence_position = 0
        matched_steps = 0
        
        for rxn in reactions:
            current_group = self.protection_sequence[sequence_position % len(self.protection_sequence)]
            
            if self.is_protection_reaction(rxn, current_group):
                matched_steps += 1
                sequence_position += 1
            elif self.is_deprotection_reaction(rxn, current_group):
                matched_steps += 1
                sequence_position += 1
        
        return matched_steps
    
    def is_protection_reaction(self, rxn, protecting_group):
        """Check if reaction involves protection with the specified group."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # For protection: free amine in reactants, protected group in products
        if protecting_group == "free_amine":
            return False  # free_amine is not a protection reaction
            
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Check for free amine in reactants
            has_free_amine_reactant = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.protecting_group_patterns["free_amine"]))
                for mol in reactant_mols
            )
            
            # Check for protecting group in products
            has_protected_product = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.protecting_group_patterns[protecting_group]))
                for mol in product_mols
            )
            
            return has_free_amine_reactant and has_protected_product
            
        except:
            return False
    
    def is_deprotection_reaction(self, rxn, protecting_group):
        """Check if reaction involves deprotection of the specified group."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # For deprotection to free amine
        if protecting_group == "free_amine":
            return self.has_deprotection_to_free_amine(reactants, products)
            
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Check for protecting group in reactants
            has_protected_reactant = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.protecting_group_patterns[protecting_group]))
                for mol in reactant_mols
            )
            
            # Check for free amine in products (deprotection case)
            has_free_amine_product = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.protecting_group_patterns["free_amine"]))
                for mol in product_mols
            )
            
            return has_protected_reactant and has_free_amine_product
            
        except:
            return False
    
    def has_deprotection_to_free_amine(self, reactants, products):
        """Check if any protecting group is removed to give free amine."""
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Check if any protecting group is present in reactants
            has_any_protection = False
            for group_name, pattern in self.protecting_group_patterns.items():
                if group_name != "free_amine":
                    if any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for mol in reactant_mols):
                        has_any_protection = True
                        break
            
            # Check for free amine in products
            has_free_amine_product = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.protecting_group_patterns["free_amine"]))
                for mol in product_mols
            )
            
            return has_any_protection and has_free_amine_product
            
        except:
            return False
