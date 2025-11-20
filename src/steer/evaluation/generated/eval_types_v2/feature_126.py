"""Generated evaluation code for: Late stage triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTriazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage triazole ring formation.
    Checks if a triazole ring (c1nnc[nH]1) is formed within a specified number 
    of steps from the end of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.step_from_end = config["parameters"]["step_from_end"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Better score for later formation (closer to end)
            return max(0, 10 - (x * 10))
        else:
            # For early timing, better score for earlier formation
            return min(10, x * 10)
    
    def hit_condition(self, d):
        """
        Check if triazole ring formation occurs in this reaction step.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".") if p.strip()]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count triazole rings in reactants and products
            reactant_triazole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in reactants
            )
            
            product_triazole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in products
            )
            
            # Ring formation occurs if product has more triazole rings than reactants
            return product_triazole_count > reactant_triazole_count
            
        except Exception:
            return False
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Find the depth at which triazole formation occurs, measured from the end.
        """
        def get_depth_from_end(node, current_depth=0):
            # If this is a leaf node (no children), return current depth
            if not node.get("children"):
                if self.hit_condition(node):
                    return current_depth
                return -1
            
            # Check current node
            if self.hit_condition(node):
                return current_depth
            
            # Recursively check children
            for child in node.get("children", []):
                result = get_depth_from_end(child, current_depth + 1)
                if result >= 0:
                    return result
            
            return -1
        
        depth_from_end = get_depth_from_end(d)
        
        if depth_from_end >= 0:
            # Check if it meets the timing requirement
            if self.timing == "late" and depth_from_end <= self.step_from_end:
                return True, depth_from_end
            elif self.timing == "early":
                return True, depth_from_end
        
        return False, -1
