"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagyPyrazoleFormation(BaseScoring):
    """
    Evaluates late-stage pyrazole ring formation in synthesis routes.
    
    Detects when a pyrazole ring (c1nn[cH][cH]1) is formed during synthesis,
    with preference for formation occurring later in the route (closer to the target).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "c1nn[cH][cH]1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Later formation (higher x) gets higher score for late-stage preference.
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            return x * 10  # Later is better, score 0-10
        elif self.timing == "early":
            return (1 - x) * 10  # Earlier is better
        else:
            return 5  # Neutral scoring if timing preference not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves pyrazole ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains pyrazole ring
            product_has_pyrazole = product.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_pyrazole:
                return False
            
            # For ring formation, check that reactants don't have the complete pyrazole
            if self.direction == "formation":
                reactants_have_pyrazole = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
                return not reactants_have_pyrazole
            
            # For ring breaking, check that reactants have the pyrazole
            elif self.direction == "breaking":
                reactants_have_pyrazole = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
                return reactants_have_pyrazole
            
            return True
            
        except Exception:
            return False
