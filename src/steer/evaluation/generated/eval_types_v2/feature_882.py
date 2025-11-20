"""Generated evaluation code for: Global acetate deprotection as final step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GlobalAcetateDeprotectionFinal(BaseScoring):
    """
    Checks if global acetate deprotection occurs as the final step in the synthesis route.
    Verifies that multiple acetate groups are removed simultaneously in the last reaction.
    """
    
    def __init__(self, config: Dict):
        self.min_simultaneous_groups = config["parameters"].get("simultaneous_groups", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        elif x == 1.0:  # Final step (depth fraction = 1.0)
            return 10
        else:
            return max(0, 10 - (1.0 - x) * 20)  # Penalize if not final step
    
    def hit_condition(self, d):
        """Check if this reaction performs global acetate deprotection"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactant_smiles, product_smiles = rxn_smiles.split(">>")
            
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if not reactant_mol or not product_mol:
                return False
            
            # Define acetate pattern (acetyl ester)
            acetate_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[CH3].[O:3][C:1](=[O:2])[CH3]")
            
            # Count acetate groups in reactant and product
            reactant_acetates = len(reactant_mol.GetSubstructMatches(acetate_pattern))
            product_acetates = len(product_mol.GetSubstructMatches(acetate_pattern))
            
            # Check if multiple acetate groups were removed
            acetates_removed = reactant_acetates - product_acetates
            
            # Verify this is a deprotection (acetates removed, not added)
            # and meets minimum simultaneous deprotection requirement
            if acetates_removed >= self.min_simultaneous_groups:
                # Additional check: ensure corresponding hydroxyl groups appear
                oh_pattern = Chem.MolFromSmarts("[OH]")
                reactant_oh = len(reactant_mol.GetSubstructMatches(oh_pattern))
                product_oh = len(product_mol.GetSubstructMatches(oh_pattern))
                oh_gained = product_oh - reactant_oh
                
                # Should gain approximately same number of OH groups as acetates lost
                return abs(oh_gained - acetates_removed) <= 1
            
            return False
            
        except Exception:
            return False
