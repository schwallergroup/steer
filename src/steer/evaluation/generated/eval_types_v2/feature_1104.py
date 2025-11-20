"""Generated evaluation code for: Late stage piperazinone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when a specific ring is formed.
    Rewards late-stage ring formation over early-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config.get("timing", "late")  # "late" or "early"
        self.direction = config.get("direction", "formation")  # "formation" or "breaking"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Ring formation/breaking doesn't happen
        
        if self.timing == "late":
            # Reward later formation (higher depth fraction = better score)
            return x * 10
        else:  # early timing
            # Reward earlier formation (lower depth fraction = better score)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves the target ring formation/breaking"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not products or not all(reactants):
                return False
            
            # Check for ring presence in products and reactants
            ring_in_products = products.HasSubstructMatch(self.ring_pattern)
            ring_in_reactants = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants if r)
            
            if self.direction == "formation":
                # Ring formation: ring absent in reactants, present in products
                return not ring_in_reactants and ring_in_products
            else:  # breaking
                # Ring breaking: ring present in reactants, absent in products
                return ring_in_reactants and not ring_in_products
                
        except Exception:
            return False
