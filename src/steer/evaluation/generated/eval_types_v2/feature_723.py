"""Generated evaluation code for: Early ester protection before bromodesilylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyEsterProtection(BaseScoring):
    """
    Evaluates whether ester protection of carboxylic acid occurs before bromodesilylation.
    Scores based on the timing of ester formation relative to Si-Br bond formation.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        self.protection_type = config.get("protection_type", "ester")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ester protection doesn't happen before bromodesilylation
        else:
            return 1 - x  # Earlier protection is better
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves ester protection of carboxylic acid"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[1].split(".")]
            
            # Pattern for carboxylic acid
            carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
            # Pattern for ester 
            ester_pattern = Chem.MolFromSmarts("[CX3](=O)[O][CX4]")
            
            # Check if reaction converts carboxylic acid to ester
            has_acid_reactant = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactants if mol)
            has_ester_product = any(mol.HasSubstructMatch(ester_pattern) for mol in products if mol)
            
            return has_acid_reactant and has_ester_product
            
        except Exception:
            return False
            
    def get_bromodesilylation_depth(self, route_dict) -> int:
        """Find the depth at which bromodesilylation occurs"""
        queue = [(route_dict, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            
            if self.is_bromodesilylation(node):
                return depth
                
            # Add children to queue
            children = node.get("children", [])
            for child in children:
                queue.append((child, depth + 1))
                
        return -1  # Bromodesilylation not found
        
    def is_bromodesilylation(self, d) -> bool:
        """Check if this reaction is a bromodesilylation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[1].split(".")]
            
            # Pattern for Si-H or Si-alkyl bond (being replaced)
            silicon_pattern = Chem.MolFromSmarts("[Si]")
            # Pattern for C-Br bond (being formed)
            bromide_pattern = Chem.MolFromSmarts("[C][Br]")
            
            has_silicon_reactant = any(mol.HasSubstructMatch(silicon_pattern) for mol in reactants if mol)
            has_bromide_product = any(mol.HasSubstructMatch(bromide_pattern) for mol in products if mol)
            
            return has_silicon_reactant and has_bromide_product
            
        except Exception:
            return False
            
    def condition_depth(self, route_dict) -> Tuple[bool, int]:
        """
        Override to implement custom logic comparing ester protection depth 
        to bromodesilylation depth
        """
        ester_depth = -1
        bromodesilylation_depth = self.get_bromodesilylation_depth(route_dict)
        
        # Find ester protection depth using BFS
        queue = [(route_dict, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            
            if self.hit_condition(node):
                ester_depth = depth
                break
                
            children = node.get("children", [])
            for child in children:
                queue.append((child, depth + 1))
        
        # Check if ester protection occurs before bromodesilylation
        if ester_depth >= 0 and bromodesilylation_depth >= 0:
            condition_met = ester_depth < bromodesilylation_depth
            # Return relative timing as fraction
            total_depth = max(ester_depth, bromodesilylation_depth)
            if total_depth > 0:
                depth_fraction = (bromodesilylation_depth - ester_depth) / total_depth
            else:
                depth_fraction = 0
            return condition_met, depth_fraction
        else:
            return False, -1
