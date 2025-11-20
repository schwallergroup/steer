"""Generated evaluation code for: Unstable N-carboxy protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class UnstableNCarboxyProtectingGroup(BaseScoring):
    """
    Evaluates routes that carry unstable N-carboxy protecting groups through multiple synthetic steps.
    Detects the presence of N-carboxy groups and penalizes routes where they persist for too many steps.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group_smarts = config["parameters"]["protecting_group_smarts"]
        self.steps_carried = config["parameters"]["steps_carried"]
        self.atom_protected = config["parameters"]["atom_protected"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protecting group not found or not carried long enough
        
        # Penalize routes where unstable protecting group is carried too long
        # Higher penalty for longer persistence
        penalty = min(x * 2, 10)  # Scale to 0-10 range
        return penalty
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves an unstable N-carboxy protecting group
        that persists through multiple steps.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create pattern for N-carboxy protecting group
            pattern = Chem.MolFromSmarts(self.protecting_group_smarts)
            if pattern is None:
                return False
            
            # Check if N-carboxy group is present in both reactants and products
            # indicating it's being carried through the reaction
            reactant_has_group = any(mol.HasSubstructMatch(pattern) for mol in reactants)
            product_has_group = any(mol.HasSubstructMatch(pattern) for mol in products)
            
            # Return True if the protecting group persists through the reaction
            return reactant_has_group and product_has_group
            
        except Exception:
            return False
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Custom implementation to count consecutive steps where protecting group is carried.
        """
        def count_protecting_group_persistence(node, depth=0):
            if self.hit_condition(node):
                # Found protecting group, check children for persistence
                max_persistence = 1
                
                for child in node.get("children", []):
                    child_persistence = count_protecting_group_persistence(child, depth + 1)
                    if child_persistence > 0:
                        max_persistence = max(max_persistence, child_persistence + 1)
                
                return max_persistence
            else:
                # No protecting group found, check children independently
                max_persistence = 0
                for child in node.get("children", []):
                    child_persistence = count_protecting_group_persistence(child, depth + 1)
                    max_persistence = max(max_persistence, child_persistence)
                
                return max_persistence
        
        steps_carried = count_protecting_group_persistence(d)
        condition_met = steps_carried >= self.steps_carried
        
        return condition_met, steps_carried
