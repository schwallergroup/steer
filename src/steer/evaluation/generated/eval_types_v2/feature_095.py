"""Generated evaluation code for: Late stage cyclopropane formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation.
    Checks if a specified ring structure is formed late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late", "early", or "any"
        self.direction = config["parameters"]["direction"]  # "formation" or "breaking"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Late-stage formation is better (higher score for higher depth fraction)
        elif self.timing == "early":
            return x  # Early-stage formation is better (higher score for lower depth fraction)
        else:  # timing == "any"
            return 1  # Just care that it happens, not when
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves the specified ring formation/breaking"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            products = [p for p in products if p is not None]
            reactants = [r for r in reactants if r is not None]
            
            # Count ring matches in reactants and products
            reactant_matches = sum(len(r.GetSubstructMatches(self.ring_pattern)) for r in reactants)
            product_matches = sum(len(p.GetSubstructMatches(self.ring_pattern)) for p in products)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_matches > reactant_matches
            elif self.direction == "breaking":
                # Ring breaking: fewer rings in products than reactants
                return reactant_matches > product_matches
            else:
                # Any change in ring count
                return reactant_matches != product_matches
                
        except Exception:
            return False
