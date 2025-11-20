"""Generated evaluation code for: Acetal protecting group strategy for aldehyde"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetalProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes based on acetal protecting group strategy for aldehydes.
    Checks if aldehydes are protected as acetals at the specified depth and deprotected later.
    """
    
    def __init__(self, config: Dict):
        self.protection_depth = config["parameters"]["protection_depth"]
        self.deprotection_depth = config["parameters"]["deprotection_depth"]
        self.aldehyde_pattern = Chem.MolFromSmarts("[CH1]=O")  # Aldehyde pattern
        self.acetal_pattern = Chem.MolFromSmarts("[CH1](O[CH3])(O[CH3])")  # Dimethyl acetal pattern
        self.current_mode = "protection"  # Track whether we're looking for protection or deprotection
        
    def route_scoring(self, protection_depth, deprotection_depth) -> float:
        """
        Score based on whether protection and deprotection occur at expected depths.
        Returns higher score (closer to 1) when depths match expectations.
        """
        if protection_depth < 0 or deprotection_depth < 0:
            return 0  # Strategy not implemented
        
        protection_score = 1 - abs(protection_depth - self.protection_depth) / 10
        deprotection_score = 1 - abs(deprotection_depth - self.deprotection_depth) / 10
        
        # Both events should occur, with protection before deprotection
        if protection_depth > deprotection_depth:
            return 0  # Invalid order
        
        return (protection_score + deprotection_score) / 2
    
    def condition_depth(self, d) -> Tuple[bool, Tuple[int, int]]:
        """
        Find depths of both acetal protection and deprotection events.
        Returns (condition_met, (protection_depth, deprotection_depth))
        """
        from collections import deque
        
        queue = deque([(d, 0)])
        protection_depth = -1
        deprotection_depth = -1
        
        while queue:
            node, depth = queue.popleft()
            
            if "children" in node:
                # Check current reaction for protection/deprotection
                if self.is_acetal_protection(node):
                    protection_depth = depth
                elif self.is_acetal_deprotection(node):
                    deprotection_depth = depth
                
                # Continue BFS
                for child in node["children"]:
                    queue.append((child, depth + 1))
        
        condition_met = protection_depth >= 0 and deprotection_depth >= 0
        return condition_met, (protection_depth, deprotection_depth)
    
    def is_acetal_protection(self, d) -> bool:
        """Check if reaction converts aldehyde to acetal (protection)"""
        if "metadata" not in d or "mapped_reaction_smiles" not in d["metadata"]:
            return False
        
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants + products):
                return False
            
            # Check if any reactant has aldehyde and any product has acetal
            has_aldehyde_reactant = any(mol.HasSubstructMatch(self.aldehyde_pattern) for mol in reactants)
            has_acetal_product = any(mol.HasSubstructMatch(self.acetal_pattern) for mol in products)
            
            return has_aldehyde_reactant and has_acetal_product
            
        except:
            return False
    
    def is_acetal_deprotection(self, d) -> bool:
        """Check if reaction converts acetal to aldehyde (deprotection)"""
        if "metadata" not in d or "mapped_reaction_smiles" not in d["metadata"]:
            return False
        
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants + products):
                return False
            
            # Check if any reactant has acetal and any product has aldehyde
            has_acetal_reactant = any(mol.HasSubstructMatch(self.acetal_pattern) for mol in reactants)
            has_aldehyde_product = any(mol.HasSubstructMatch(self.aldehyde_pattern) for mol in products)
            
            return has_acetal_reactant and has_aldehyde_product
            
        except:
            return False
