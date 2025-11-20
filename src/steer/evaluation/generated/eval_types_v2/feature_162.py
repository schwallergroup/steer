"""Generated evaluation code for: Late stage cyclopropanation as final step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring formation occurs at a specified timing in the synthesis route.
    Checks if cyclopropyl ring formation happens as the final step (depth 0).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1" for cyclopropyl
        self.timing = config["parameters"]["timing"]  # "late"
        self.step_position = config["parameters"]["step_position"]  # "final"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        For final step requirement, only depth 0 gets full score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.step_position == "final":
            # Only final step (depth 0) gets full score
            if x == 0:
                return 10
            else:
                return 0
        elif self.timing == "late":
            # Later is better - higher score for lower depth
            return max(0, 10 * (1 - x))
        else:
            # Default: any occurrence gets some score
            return 5 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves formation of the target ring structure.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count target rings in reactants and products
            reactant_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                    for mol in reactants)
            product_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                   for mol in products)
            
            # Ring formation: more rings in products than reactants
            return product_ring_count > reactant_ring_count
            
        except Exception:
            return False
