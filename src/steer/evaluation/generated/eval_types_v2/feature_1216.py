"""Generated evaluation code for: Late stage pyrazolopyrimidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazolopyrimidineRingFormation(BaseScoring):
    """
    Evaluates the timing of pyrazolopyrimidine ring formation in synthesis routes.
    Checks if the fused bicyclic pyrazolopyrimidine system is formed at the specified timing.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config["timing"]  # "early", "late", etc.
        self.direction = config["direction"]  # "formation" or "break"
        self.target_depth = self._get_target_depth()
        
    def _get_target_depth(self):
        """Convert timing to target depth fraction"""
        timing_map = {
            "early": 0.2,
            "mid": 0.5,
            "late": 0.8
        }
        return timing_map.get(self.timing, 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation/break doesn't happen
        
        # Score based on how close the actual depth is to target depth
        depth_diff = abs(x - self.target_depth)
        # Convert to 0-10 scale where closer to target = higher score
        return max(0, 10 * (1 - depth_diff * 2))
    
    def hit_condition(self, d):
        """Check if pyrazolopyrimidine ring formation/break occurs in this reaction"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create pattern for pyrazolopyrimidine
            pattern = Chem.MolFromSmarts(self.ring_smarts)
            if pattern is None:
                return False
            
            # Count occurrences in reactants and products
            reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in reactants)
            product_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in products)
            
            if self.direction == "formation":
                # Ring formation: fewer in reactants, more in products
                return product_matches > reactant_matches
            elif self.direction == "break":
                # Ring breaking: more in reactants, fewer in products
                return reactant_matches > product_matches
            
        except Exception:
            return False
            
        return False
