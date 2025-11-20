"""Generated evaluation code for: Late stage pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring formation occurs at a late stage in the synthesis.
    Uses SMARTS pattern matching to detect when the target ring is formed.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early", "late", etc.
        self.direction = config["parameters"]["direction"]  # "formation" or "breaking"
        
        # Convert timing preference to scoring parameters
        if self.timing == "late":
            self.preferred_depth_fraction = 0.8  # Prefer reactions near end of route
        elif self.timing == "early": 
            self.preferred_depth_fraction = 0.2  # Prefer reactions near start of route
        else:
            self.preferred_depth_fraction = 0.5  # Middle of route
    
    def route_scoring(self, x: float) -> float:
        """Convert depth fraction to 0-10 score based on timing preference"""
        if x < 0:
            return 0  # Ring formation/breaking doesn't occur
        
        # Score based on how close the timing is to preference
        distance_from_preferred = abs(x - self.preferred_depth_fraction)
        
        # Convert to 0-10 scale (closer to preferred = higher score)
        score = 10 * (1 - distance_from_preferred)
        return max(0, score)
    
    def hit_condition(self, d: Dict) -> bool:
        """Check if this reaction involves the target ring formation/breaking"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create SMARTS pattern for ring detection
            ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if ring_pattern is None:
                return False
            
            # Count ring occurrences in reactants and products
            reactant_matches = sum(len(mol.GetSubstructMatches(ring_pattern)) 
                                 for mol in reactants)
            product_matches = sum(len(mol.GetSubstructMatches(ring_pattern)) 
                                for mol in products)
            
            # Check if ring formation or breaking occurred
            if self.direction == "formation":
                return product_matches > reactant_matches
            elif self.direction == "breaking":
                return reactant_matches > product_matches
            
            return False
            
        except Exception:
            return False
