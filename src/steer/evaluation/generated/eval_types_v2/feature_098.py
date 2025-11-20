"""Generated evaluation code for: Late isoxazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IsoxazoleFormationDepth(BaseScoring):
    """
    Evaluates the depth at which isoxazole ring formation occurs in a synthesis route.
    Rewards late-stage isoxazole formation as a key coupling step between fragments.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        else:
            # Late-stage formation is better (higher depth fraction is rewarded)
            return x * 10  # Convert depth fraction to 0-10 score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves isoxazole ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Count isoxazole rings in product
            product_rings = len(product.GetSubstructMatches(self.ring_pattern))
            
            # Count total isoxazole rings in all reactants
            reactant_rings = sum(len(r.GetSubstructMatches(self.ring_pattern)) for r in reactants)
            
            # Check for ring formation (more rings in product than in reactants)
            if self.direction == "formation":
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                return product_rings < reactant_rings
            
        except Exception:
            return False
        
        return False
