"""Generated evaluation code for: Early stage piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPiperidineFormation(BaseScoring):
    """
    Evaluates whether piperidine ring formation occurs early in the synthesis route.
    Checks for the formation of piperidine rings (C1CCNCC1) and rewards earlier occurrence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Earlier formation is better (higher score)
            elif self.timing == "late":
                return x  # Later formation is better
            else:
                return 0.5  # Neutral if timing not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if piperidine ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        # Parse product and reactants
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if piperidine ring is formed (present in product but not in reactants)
            product_has_ring = product.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_ring:
                return False
                
            # Check if any reactant already has the piperidine ring
            reactants_have_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            
            # Ring formation occurs if product has ring but reactants don't
            if self.direction == "formation":
                return product_has_ring and not reactants_have_ring
            elif self.direction == "breaking":
                return not product_has_ring and reactants_have_ring
            else:
                return False
                
        except Exception:
            return False
