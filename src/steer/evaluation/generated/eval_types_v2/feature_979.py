"""Generated evaluation code for: Sequential benzyl protection-deprotection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectionDeprotectionStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential benzyl protection-deprotection strategy.
    Checks if benzyl protection occurs at specified depth followed by deprotection at another depth.
    """
    
    def __init__(self, config):
        self.protecting_step_depth = config["parameters"]["protecting_step_depth"]
        self.deprotecting_step_depth = config["parameters"]["deprotecting_step_depth"]
        # Benzyl protection patterns
        self.benzyl_ether_pattern = Chem.MolFromSmarts("COCc1ccccc1")  # Benzyl ether
        self.benzyl_ester_pattern = Chem.MolFromSmarts("C(=O)OCc1ccccc1")  # Benzyl ester
        self.phenol_pattern = Chem.MolFromSmarts("c1ccccc1O")  # Free phenol
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")  # Free carboxylic acid
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        protection_found = False
        deprotection_found = False
        protection_depth = -1
        deprotection_depth = -1
        
        # Check each reaction for protection/deprotection
        for i, rxn in enumerate(reactions):
            current_depth = i + 1
            
            if self.is_benzyl_protection(rxn):
                protection_found = True
                protection_depth = current_depth
            
            if self.is_benzyl_deprotection(rxn):
                deprotection_found = True
                deprotection_depth = current_depth
        
        # Check if both operations occur at specified depths
        protection_at_target = (protection_depth == self.protecting_step_depth)
        deprotection_at_target = (deprotection_depth == self.deprotecting_step_depth)
        
        # Sequential requirement: protection should occur before deprotection
        sequential_order = protection_found and deprotection_found and (protection_depth < deprotection_depth)
        
        condition_met = protection_at_target and deprotection_at_target and sequential_order
        
        return condition_met, total_reactions
    
    def is_benzyl_protection(self, rxn):
        """Check if reaction involves benzyl protection (formation of benzyl ether/ester)"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Check if benzyl groups increase from reactants to products
            reactant_benzyl_count = sum(self.count_benzyl_groups(mol) for mol in reactants if mol)
            product_benzyl_count = sum(self.count_benzyl_groups(mol) for mol in products if mol)
            
            # Also check if free OH/COOH decreases (being protected)
            reactant_free_groups = sum(self.count_free_groups(mol) for mol in reactants if mol)
            product_free_groups = sum(self.count_free_groups(mol) for mol in products if mol)
            
            return (product_benzyl_count > reactant_benzyl_count) and (product_free_groups < reactant_free_groups)
            
        except:
            return False
    
    def is_benzyl_deprotection(self, rxn):
        """Check if reaction involves benzyl deprotection (removal of benzyl ether/ester)"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Check if benzyl groups decrease from reactants to products
            reactant_benzyl_count = sum(self.count_benzyl_groups(mol) for mol in reactants if mol)
            product_benzyl_count = sum(self.count_benzyl_groups(mol) for mol in products if mol)
            
            # Also check if free OH/COOH increases (being deprotected)
            reactant_free_groups = sum(self.count_free_groups(mol) for mol in reactants if mol)
            product_free_groups = sum(self.count_free_groups(mol) for mol in products if mol)
            
            return (reactant_benzyl_count > product_benzyl_count) and (product_free_groups > reactant_free_groups)
            
        except:
            return False
    
    def count_benzyl_groups(self, mol):
        """Count benzyl ether and ester groups in molecule"""
        if not mol:
            return 0
        benzyl_ether_matches = mol.GetSubstructMatches(self.benzyl_ether_pattern)
        benzyl_ester_matches = mol.GetSubstructMatches(self.benzyl_ester_pattern)
        return len(benzyl_ether_matches) + len(benzyl_ester_matches)
    
    def count_free_groups(self, mol):
        """Count free phenol and carboxylic acid groups"""
        if not mol:
            return 0
        phenol_matches = mol.GetSubstructMatches(self.phenol_pattern)
        carboxylic_matches = mol.GetSubstructMatches(self.carboxylic_acid_pattern)
        return len(phenol_matches) + len(carboxylic_matches)
