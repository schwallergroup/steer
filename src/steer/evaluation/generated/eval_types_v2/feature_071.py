"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """Evaluates whether a specific ring is formed late in the synthesis route.
    
    Searches for ring formation reactions where the target ring structure appears
    in the product but not in the reactants, indicating ring formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        
        # Convert timing preference to target depth
        if self.timing == "late":
            self.prefer_late = True
        else:
            self.prefer_late = False
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.prefer_late:
            # Higher scores for later formation (closer to 1.0)
            return x * 10
        else:
            # Higher scores for earlier formation (closer to 0.0)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node contains the target ring formation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Create pattern molecule for substructure matching
            ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not ring_pattern:
                return False
            
            if self.direction == "formation":
                # Ring formation: pattern present in product but absent in all reactants
                product_has_ring = product.HasSubstructMatch(ring_pattern)
                reactants_have_ring = any(r.HasSubstructMatch(ring_pattern) for r in reactants)
                
                return product_has_ring and not reactants_have_ring
            
            else:  # direction == "break"
                # Ring breaking: pattern present in reactants but absent in product
                product_has_ring = product.HasSubstructMatch(ring_pattern)
                reactants_have_ring = any(r.HasSubstructMatch(ring_pattern) for r in reactants)
                
                return not product_has_ring and reactants_have_ring
                
        except (KeyError, ValueError, AttributeError):
            return False
