"""Generated evaluation code for: Early lactone ring opening strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyLactoneRingOpening(BaseScoring):
    """
    Evaluates whether lactone ring opening occurs early in the synthesis route.
    Detects lactone ring breaking reactions and scores based on their timing.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        
        # Compile SMARTS pattern for lactone detection
        self.lactone_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Early timing means we want low depth values (closer to 0)
        self.prefer_early = (self.timing == "early")
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        For early timing: lower depth = higher score
        """
        if x < 0:
            return 0  # Ring opening doesn't occur
        
        if self.prefer_early:
            # Early is better: score decreases with depth
            return max(0, 10 * (1 - x))
        else:
            # Late is better: score increases with depth  
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves lactone ring opening.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            
            # Parse product and reactants
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains lactone ring
            has_lactone_in_product = product.HasSubstructMatch(self.lactone_pattern)
            
            if not has_lactone_in_product:
                return False
            
            # Check if lactone ring is broken in any reactant
            # (lactone present in product but fragmented in reactants)
            lactone_intact_in_reactants = any(r.HasSubstructMatch(self.lactone_pattern) for r in reactants)
            
            # Ring opening: lactone intact in product, broken in reactants
            if self.direction == "breaking":
                return has_lactone_in_product and not lactone_intact_in_reactants
            
            return False
            
        except Exception:
            return False
