"""Generated evaluation code for: Late pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring is formed late in the synthesis route.
    Checks for the formation of a pyridine ring (or other specified ring) and
    scores based on how late in the synthesis this formation occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "breaking"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation/breaking doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better, so invert the depth fraction
        else:  # early timing
            return x  # Earlier is better, so use depth fraction directly
    
    def hit_condition(self, d):
        """
        Check if the specified ring formation/breaking occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        products = Chem.MolFromSmiles(rxn_parts[0])
        reactants_smiles = rxn_parts[1].split(".")
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
        
        if not products or not reactants:
            return False
            
        # Create pattern for ring detection
        ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if not ring_pattern:
            return False
            
        # Check for ring presence in products and reactants
        ring_in_products = products.HasSubstructMatch(ring_pattern)
        ring_in_reactants = any(r.HasSubstructMatch(ring_pattern) for r in reactants if r)
        
        if self.direction == "formation":
            # Ring formation: ring present in products but not in reactants
            return ring_in_products and not ring_in_reactants
        else:  # breaking
            # Ring breaking: ring present in reactants but not in products
            return not ring_in_products and ring_in_reactants
