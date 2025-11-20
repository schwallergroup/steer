"""Generated evaluation code for: Late stage beta-lactam ring closure"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BetaLactamRingClosure(BaseScoring):
    """
    Evaluates the timing of beta-lactam ring formation in synthesis routes.
    Detects when a four-membered beta-lactam ring is formed and scores based on
    whether it occurs at the desired timing (late stage).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        
        # Convert timing preference to target depth range
        if self.timing == "late":
            self.preferred_min_depth = 0.6  # Late stage means > 60% through synthesis
        elif self.timing == "early":
            self.preferred_max_depth = 0.4  # Early stage means < 40% through synthesis
        else:  # middle
            self.preferred_min_depth = 0.3
            self.preferred_max_depth = 0.7

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Higher score for later formation (x closer to 1)
            if x >= self.preferred_min_depth:
                return 10.0  # Perfect late stage timing
            else:
                return max(0, 10.0 * (x / self.preferred_min_depth))
        elif self.timing == "early":
            # Higher score for earlier formation (x closer to 0)
            if x <= self.preferred_max_depth:
                return 10.0  # Perfect early stage timing
            else:
                return max(0, 10.0 * (1 - x) / (1 - self.preferred_max_depth))
        else:  # middle timing
            if self.preferred_min_depth <= x <= self.preferred_max_depth:
                return 10.0  # Perfect middle stage timing
            elif x < self.preferred_min_depth:
                return max(0, 10.0 * x / self.preferred_min_depth)
            else:
                return max(0, 10.0 * (1 - x) / (1 - self.preferred_max_depth))

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves beta-lactam ring formation.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if any(mol is None for mol in reactants + products):
                return False
            
            # Create pattern for beta-lactam ring
            pattern = Chem.MolFromSmarts(self.ring_smarts)
            if pattern is None:
                return False
            
            if self.direction == "formation":
                # Check if beta-lactam ring is absent in reactants but present in products
                reactant_has_ring = any(mol.HasSubstructMatch(pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(pattern) for mol in products)
                
                return not reactant_has_ring and product_has_ring
                
            elif self.direction == "breaking":
                # Check if beta-lactam ring is present in reactants but absent in products
                reactant_has_ring = any(mol.HasSubstructMatch(pattern) for mol in reactants)
                product_has_ring = any(mol.HasSubstructMatch(pattern) for mol in products)
                
                return reactant_has_ring and not product_has_ring
                
        except Exception:
            return False
            
        return False
