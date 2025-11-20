"""Generated evaluation code for: Multiple orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleOrthogonalProtectingGroups(MultiRxnCondBase):
    """
    Evaluates routes based on the use of multiple orthogonal protecting groups.
    Checks if the route employs the specified protecting groups with different
    deprotection conditions for selective removal.
    """
    
    def __init__(self, config):
        self.required_groups = config["protecting_groups"]
        self.orthogonal_count = config["orthogonal_count"]
        self.strategy_type = config["strategy_type"]
        
        # Define SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "benzyl": "[CH2]c1ccccc1",  # Benzyl group
            "nosyl": "S(=O)(=O)c1ccc(cc1)[N+](=O)[O-]",  # Nosyl (4-nitrobenzenesulfonyl)
            "TBS": "[Si](C)(C)C(C)(C)C",  # tert-Butyldimethylsilyl
            "formate": "C(=O)[H]",  # Formate ester
        }
    
    def condition_depth(self, d):
        """Check if the route uses the required orthogonal protecting groups"""
        reactions = self.get_rxns(d)
        detected_groups = set()
        
        for rxn in reactions:
            for group_name in self.required_groups:
                if self.detect_protecting_group_usage(rxn, group_name):
                    detected_groups.add(group_name)
        
        # Check if we found the required number of orthogonal protecting groups
        condition_met = len(detected_groups) >= self.orthogonal_count
        
        # For multiple orthogonal strategy, we want all required groups present
        if self.strategy_type == "multiple_orthogonal":
            condition_met = detected_groups >= set(self.required_groups)
        
        return condition_met, len(reactions)
    
    def detect_protecting_group_usage(self, rxn, group_name):
        """Detect if a specific protecting group is used in the reaction"""
        if group_name not in self.protecting_group_patterns:
            return False
            
        pattern = self.protecting_group_patterns[group_name]
        
        try:
            # Parse reaction SMILES
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for protection reaction (group appears in product but not reactants)
            protection_detected = self._check_protection_reaction(reactants, products, pattern)
            
            # Check for deprotection reaction (group disappears from reactants to products)
            deprotection_detected = self._check_deprotection_reaction(reactants, products, pattern)
            
            return protection_detected or deprotection_detected
            
        except Exception:
            return False
    
    def _check_protection_reaction(self, reactants, products, pattern):
        """Check if protecting group is introduced in the reaction"""
        try:
            mol_pattern = Chem.MolFromSmarts(pattern)
            if mol_pattern is None:
                return False
            
            # Count occurrences in reactants
            reactant_count = 0
            for smiles in reactants.split("."):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    reactant_count += len(mol.GetSubstructMatches(mol_pattern))
            
            # Count occurrences in products
            product_count = 0
            for smiles in products.split("."):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    product_count += len(mol.GetSubstructMatches(mol_pattern))
            
            # Protection: more occurrences in products than reactants
            return product_count > reactant_count
            
        except Exception:
            return False
    
    def _check_deprotection_reaction(self, reactants, products, pattern):
        """Check if protecting group is removed in the reaction"""
        try:
            mol_pattern = Chem.MolFromSmarts(pattern)
            if mol_pattern is None:
                return False
            
            # Count occurrences in reactants
            reactant_count = 0
            for smiles in reactants.split("."):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    reactant_count += len(mol.GetSubstructMatches(mol_pattern))
            
            # Count occurrences in products
            product_count = 0
            for smiles in products.split("."):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    product_count += len(mol.GetSubstructMatches(mol_pattern))
            
            # Deprotection: fewer occurrences in products than reactants
            return reactant_count > product_count
            
        except Exception:
            return False
