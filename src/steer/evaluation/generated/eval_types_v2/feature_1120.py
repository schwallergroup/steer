"""Generated evaluation code for: Multiple carboxylic acid protecting group exchanges"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiCarboxylicAcidProtectionExchange(MultiRxnCondBase):
    """
    Evaluates routes based on multiple carboxylic acid protecting group exchanges.
    Penalizes routes that have unnecessary protection/deprotection cycles on carboxylic acids.
    """
    
    def __init__(self, config):
        self.target_protection_changes = config["parameters"]["protection_changes"]
        self.functional_group = config["parameters"]["functional_group"]
        self.expected_sequence = config["parameters"].get("sequence", [])
        
        # Define SMARTS patterns for carboxylic acid and its protected forms
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH1]")
        self.ester_patterns = {
            "benzyl_ester": Chem.MolFromSmarts("[CX3](=O)OCc1ccccc1"),
            "ethyl_ester": Chem.MolFromSmarts("[CX3](=O)OCC"),
            "butyl_ester": Chem.MolFromSmarts("[CX3](=O)OCCCC"),
            "methyl_ester": Chem.MolFromSmarts("[CX3](=O)OC"),
            "tert_butyl_ester": Chem.MolFromSmarts("[CX3](=O)OC(C)(C)C"),
            "general_ester": Chem.MolFromSmarts("[CX3](=O)O[CX4]")
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        protection_changes = self.count_protection_changes(reactions)
        
        condition = protection_changes >= self.target_protection_changes
        return condition, len(reactions)
    
    def count_protection_changes(self, reactions) -> int:
        """Count the number of carboxylic acid protection/deprotection events"""
        changes = 0
        
        for rxn_smiles in reactions:
            if self.is_protection_deprotection_reaction(rxn_smiles):
                changes += 1
        
        return changes
    
    def is_protection_deprotection_reaction(self, rxn_smiles: str) -> bool:
        """Check if reaction involves carboxylic acid protection/deprotection"""
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if any(mol is None for mol in reactants + products):
                return False
            
            # Count carboxylic acids and esters in reactants and products
            reactant_acids = sum(self.count_functional_groups(mol, self.carboxylic_acid_pattern) for mol in reactants)
            reactant_esters = sum(self.count_all_esters(mol) for mol in reactants)
            
            product_acids = sum(self.count_functional_groups(mol, self.carboxylic_acid_pattern) for mol in products)
            product_esters = sum(self.count_all_esters(mol) for mol in products)
            
            # Protection: acid -> ester (acid count decreases, ester count increases)
            # Deprotection: ester -> acid (ester count decreases, acid count increases)
            acid_change = product_acids - reactant_acids
            ester_change = product_esters - reactant_esters
            
            # Check if this represents a protection/deprotection event
            is_protection = (acid_change < 0 and ester_change > 0)
            is_deprotection = (acid_change > 0 and ester_change < 0)
            
            return is_protection or is_deprotection
            
        except Exception:
            return False
    
    def count_functional_groups(self, mol, pattern) -> int:
        """Count occurrences of a functional group pattern in molecule"""
        if mol is None or pattern is None:
            return 0
        return len(mol.GetSubstructMatches(pattern))
    
    def count_all_esters(self, mol) -> int:
        """Count all ester groups in molecule"""
        if mol is None:
            return 0
        
        total_esters = 0
        for pattern in self.ester_patterns.values():
            if pattern is not None:
                total_esters += len(mol.GetSubstructMatches(pattern))
        
        # Use general ester pattern as fallback if specific patterns don't match
        if total_esters == 0:
            general_pattern = self.ester_patterns.get("general_ester")
            if general_pattern is not None:
                total_esters = len(mol.GetSubstructMatches(general_pattern))
        
        return total_esters
    
    def route_scoring(self, x) -> float:
        """Convert protection change count to 0-10 score"""
        if x < 0:
            return 0  # No protection changes found
        
        # Higher penalty for more protection changes than target
        if x >= self.target_protection_changes:
            excess = x - self.target_protection_changes
            return min(10, 7 + excess)  # Base penalty of 7, increases with excess
        else:
            # Lower penalty for fewer protection changes
            return max(0, 5 - (self.target_protection_changes - x))
